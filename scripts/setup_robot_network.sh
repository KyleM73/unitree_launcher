#!/bin/bash
# setup_robot_network.sh — Configure network for Unitree G1 communication.
#
# Two modes:
#   ethernet — Direct Ethernet cable to robot (assign static IP, NAT, gateway)
#   wifi     — Shared WiFi network (bootstrap robot WiFi via Ethernet, then SSH over WiFi)
#
# Usage:
#   ./scripts/setup_robot_network.sh ethernet [INTERFACE]
#   ./scripts/setup_robot_network.sh wifi [SSID] [PASSWORD]
#   ./scripts/setup_robot_network.sh --help

set -euo pipefail

# ── Constants ──────────────────────────────────────────────────────────
ROBOT_MOTOR_IP="192.168.123.161"
ROBOT_PC_ETH="192.168.123.164"
ETH_SUBNET="192.168.123"
HOST_ETH_IP="${ETH_SUBNET}.100"
NETMASK="24"
PING_TIMEOUT=2
PING_COUNT=3
ROBOT_USER="unitree"

# ── Colors ─────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()   { echo -e "${RED}[error]${NC} $*"; }
header(){ echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Help ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Setup network for Unitree G1 robot communication.

Usage:
    ./scripts/setup_robot_network.sh ethernet [INTERFACE]
    ./scripts/setup_robot_network.sh wifi [SSID] [PASSWORD]

Modes:
    ethernet    Direct Ethernet connection between host and robot.
                Assigns static IP 192.168.123.100/24, sets up NAT for
                internet sharing, and configures the robot's gateway/DNS.
                Optional INTERFACE argument (e.g. en8, enp3s0) to skip
                auto-detection.

    wifi        Connect to robot over a shared WiFi network.
                If the robot isn't already on WiFi, bootstraps via Ethernet:
                unblocks WiFi radio, connects to your network, verifies SSH.
                Sets up SSH key auth for passwordless access.
                Prompts for SSID/password if not provided as arguments.

Network layout (Ethernet mode):
    Host:        192.168.123.100
    Robot PC:    192.168.123.164
    Motor board: 192.168.123.161

Network layout (WiFi mode):
    Host:        DHCP from WiFi router
    Robot PC:    DHCP from WiFi router (wlan0)
    Motor board: 192.168.123.161 (internal Ethernet, unchanged)

The robot's wireless controller and Bluetooth app are unaffected by
either mode — they use separate radios.
EOF
    exit 0
fi

# ── Detect OS ─────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)      err "Unsupported OS: $OS"; exit 1 ;;
esac

# ══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════

setup_ssh_keys() {
    local host="$1"

    # Generate key if needed
    if [[ ! -f "$HOME/.ssh/id_ed25519.pub" && ! -f "$HOME/.ssh/id_rsa.pub" ]]; then
        info "Generating SSH key pair..."
        mkdir -p "$HOME/.ssh"
        ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N "" -q
        ok "SSH key generated"
    fi

    # Check if key auth already works
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
        "$ROBOT_USER@$host" "echo ok" &>/dev/null; then
        ok "SSH key auth already configured for $host"
        return 0
    fi

    info "Copying SSH key to robot at $host ..."
    info "Enter robot password when prompted (default: 123)"
    ssh-copy-id -o StrictHostKeyChecking=no "$ROBOT_USER@$host" 2>/dev/null || true

    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
        "$ROBOT_USER@$host" "echo ok" &>/dev/null; then
        ok "SSH key auth configured for $host"
    else
        warn "Could not set up key auth — you'll need the password each time"
    fi
}

# ══════════════════════════════════════════════════════════════════════
# ETHERNET HELPERS
# ══════════════════════════════════════════════════════════════════════

get_macos_wired_ports() {
    local current_port=""
    while IFS= read -r line; do
        if [[ "$line" == "Hardware Port: "* ]]; then
            current_port="${line#Hardware Port: }"
        elif [[ "$line" == "Device: "* && -n "$current_port" ]]; then
            local dev="${line#Device: }"
            case "$current_port" in
                *Wi-Fi*|*AirPort*|*Bluetooth*|*FireWire*|*VPN*|*Bridge*|*VLAN*)
                    ;;
                *)
                    printf "%s\t%s\n" "$dev" "$current_port"
                    ;;
            esac
            current_port=""
        fi
    done < <(networksetup -listallhardwareports 2>/dev/null)
}

ensure_macos_network_service() {
    # Check if there's already an active wired interface
    while IFS=$'\t' read -r dev port_name; do
        if ifconfig "$dev" 2>/dev/null | grep -q "status: active"; then
            return
        fi
    done < <(get_macos_wired_ports)

    # No active wired interface. Check if any are missing a network service.
    local existing_services
    existing_services=$(networksetup -listallnetworkservices 2>/dev/null)
    local needs_service=""
    local needs_service_port=""

    while IFS=$'\t' read -r dev port_name; do
        local has_service=false
        if echo "$existing_services" | grep -qF "$port_name"; then
            has_service=true
        fi
        if echo "$existing_services" | grep -qF "Unitree G1"; then
            has_service=true
        fi
        if networksetup -getinfo "$port_name" &>/dev/null; then
            has_service=true
        fi
        if [[ "$has_service" == false && -z "$needs_service" ]]; then
            needs_service="$dev"
            needs_service_port="$port_name"
        fi
    done < <(get_macos_wired_ports)

    if [[ -z "$needs_service" ]]; then
        return
    fi

    warn "No network service for '$needs_service_port' ($needs_service)"
    info "Creating service 'Unitree G1' on $needs_service with static IP $HOST_ETH_IP ..."
    echo ""
    if sudo networksetup -createnetworkservice "Unitree G1" "$needs_service" && \
       sudo networksetup -setmanual "Unitree G1" "$HOST_ETH_IP" "255.255.255.0"; then
        ok "Created service 'Unitree G1' on $needs_service"
        info "Waiting for interface to activate ..."
        sleep 2
    else
        err "Could not create network service."
        echo ""
        echo "  Run these manually:"
        echo "    sudo networksetup -createnetworkservice \"Unitree G1\" $needs_service"
        echo "    sudo networksetup -setmanual \"Unitree G1\" $HOST_ETH_IP 255.255.255.0"
        echo ""
        echo "  Then re-run this script."
        exit 1
    fi
}

find_macos_ethernet() {
    local all_wired_ifaces=""
    while IFS=$'\t' read -r dev port_name; do
        all_wired_ifaces="$all_wired_ifaces $dev"
    done < <(get_macos_wired_ports)

    if [[ -z "$all_wired_ifaces" ]]; then
        return
    fi

    for iface in $all_wired_ifaces; do
        if ifconfig "$iface" 2>/dev/null | grep -q "status: active"; then
            echo "$iface"
            return
        fi
    done
}

find_linux_ethernet() {
    ip -o link show 2>/dev/null \
        | awk -F': ' '{print $2}' \
        | while read -r iface; do
            [[ "$iface" == lo ]] && continue
            [[ "$iface" == docker* ]] && continue
            [[ "$iface" == veth* ]] && continue
            [[ "$iface" == br-* ]] && continue
            [[ -d "/sys/class/net/$iface/wireless" ]] && continue
            if [[ -f "/sys/class/net/$iface/carrier" ]] && \
               [[ "$(cat "/sys/class/net/$iface/carrier" 2>/dev/null)" == "1" ]]; then
                echo "$iface"
                return
            fi
        done
}

list_macos_wired_ports() {
    while IFS= read -r line; do
        if [[ "$line" == "Hardware Port: "* ]]; then
            local current_port="${line#Hardware Port: }"
        elif [[ "$line" == "Device: "* && -n "${current_port:-}" ]]; then
            local dev="${line#Device: }"
            case "$current_port" in
                *Wi-Fi*|*AirPort*|*Bluetooth*|*FireWire*|*VPN*|*VLAN*)
                    ;;
                *)
                    local port_status
                    port_status=$(ifconfig "$dev" 2>/dev/null | awk '/status:/{print $2}')
                    printf "    %-8s  %-35s  %s\n" "$dev" "$current_port" "${port_status:-unknown}"
                    ;;
            esac
            current_port=""
        fi
    done < <(networksetup -listallhardwareports 2>/dev/null)
}

# Detect and validate an Ethernet interface. Sets IFACE variable.
detect_ethernet_interface() {
    local specified="${1:-}"

    header "Detecting Ethernet interface"

    if [[ -n "$specified" ]]; then
        IFACE="$specified"
        info "Using specified interface: $IFACE"
    else
        if [[ "$PLATFORM" == "macos" ]]; then
            ensure_macos_network_service
            IFACE=$(find_macos_ethernet)
        else
            IFACE=$(find_linux_ethernet)
        fi

        if [[ -z "${IFACE:-}" ]]; then
            err "No active Ethernet interface found (is the cable plugged in?)."
            echo ""
            if [[ "$PLATFORM" == "macos" ]]; then
                echo "  Available wired ports:"
                list_macos_wired_ports
            else
                echo "  Available interfaces:"
                ip -o link show 2>/dev/null | awk -F': ' '{print "    " $2}'
            fi
            echo ""
            echo "  Plug in the Ethernet cable and retry, or specify manually:"
            echo "    $0 ethernet <interface_name>"
            return 1
        fi
        info "Auto-detected Ethernet interface: ${BOLD}$IFACE${NC}"
    fi

    # Verify interface exists
    if [[ "$PLATFORM" == "macos" ]]; then
        if ! ifconfig "$IFACE" &>/dev/null; then
            err "Interface '$IFACE' does not exist."
            echo "  Available wired ports:"
            list_macos_wired_ports
            return 1
        fi
    else
        if ! ip link show "$IFACE" &>/dev/null; then
            err "Interface '$IFACE' does not exist."
            echo "  Available interfaces:"
            ip -o link show | awk -F': ' '{print "    " $2}'
            return 1
        fi
    fi

    ok "Interface $IFACE exists"

    # Show link info
    if [[ "$PLATFORM" == "macos" ]]; then
        local link_status media_info
        link_status=$(ifconfig "$IFACE" 2>/dev/null | awk '/status:/{print $2}')
        media_info=$(ifconfig "$IFACE" 2>/dev/null | awk '/media:/{$1=""; print}' | xargs)
        if [[ "$link_status" == "active" ]]; then
            ok "Link is active ($media_info)"
        else
            warn "Link status: $link_status"
            warn "Make sure the Ethernet cable is plugged in to this port."
        fi
    else
        local carrier speed
        carrier=$(cat "/sys/class/net/$IFACE/carrier" 2>/dev/null || echo "0")
        speed=$(cat "/sys/class/net/$IFACE/speed" 2>/dev/null || echo "?")
        if [[ "$carrier" == "1" ]]; then
            ok "Link is active (${speed} Mbps)"
        else
            warn "No carrier detected on $IFACE"
            warn "Make sure the Ethernet cable is plugged in."
        fi
    fi
}

# Remove robot-subnet IPs from other interfaces to fix routing.
remove_stale_ips() {
    local target_iface="$1"

    header "Checking for conflicting IPs on other interfaces"

    if [[ "$PLATFORM" == "macos" ]]; then
        local stale_found=false
        for other_iface in $(ifconfig -l); do
            [[ "$other_iface" == "$target_iface" ]] && continue
            local other_ips
            other_ips=$(ifconfig "$other_iface" 2>/dev/null | awk '/inet /{print $2}')
            while IFS= read -r ip; do
                if [[ "$ip" == ${ETH_SUBNET}.* ]]; then
                    warn "Found ${ETH_SUBNET}.x address $ip on $other_iface"
                    info "Removing $ip from $other_iface to fix routing ..."
                    sudo ifconfig "$other_iface" -alias "$ip" 2>/dev/null && \
                        ok "Removed $ip from $other_iface" || \
                        warn "Could not remove $ip from $other_iface"
                    stale_found=true
                fi
            done <<< "$other_ips"
        done
        if [[ "$stale_found" == false ]]; then
            ok "No conflicting IPs found"
        fi
    else
        local stale_found=false
        for other_iface in $(ip -o link show 2>/dev/null | awk -F': ' '{print $2}'); do
            [[ "$other_iface" == "$target_iface" ]] && continue
            local other_ips
            other_ips=$(ip -4 addr show "$other_iface" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1)
            while IFS= read -r ip; do
                if [[ "$ip" == ${ETH_SUBNET}.* ]]; then
                    warn "Found ${ETH_SUBNET}.x address $ip on $other_iface"
                    info "Removing $ip from $other_iface to fix routing ..."
                    sudo ip addr del "$ip/$NETMASK" dev "$other_iface" 2>/dev/null && \
                        ok "Removed $ip from $other_iface" || \
                        warn "Could not remove $ip from $other_iface"
                    stale_found=true
                fi
            done <<< "$other_ips"
        done
        if [[ "$stale_found" == false ]]; then
            ok "No conflicting IPs found"
        fi
    fi
}

# Assign static IP to the Ethernet interface.
assign_static_ip() {
    local iface="$1"

    header "Checking current IP configuration"

    local already_configured=false
    local current_ips
    if [[ "$PLATFORM" == "macos" ]]; then
        current_ips=$(ifconfig "$iface" 2>/dev/null | awk '/inet /{print $2}')
    else
        current_ips=$(ip -4 addr show "$iface" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1)
    fi

    if echo "$current_ips" | grep -q "^${HOST_ETH_IP}$"; then
        ok "Interface already has $HOST_ETH_IP"
        already_configured=true
    elif [[ -n "$current_ips" ]]; then
        info "Current IPs on $iface:"
        echo "$current_ips" | sed 's/^/    /'
    fi

    if [[ "$already_configured" == false ]]; then
        header "Assigning static IP"
        info "Setting ${BOLD}$HOST_ETH_IP/$NETMASK${NC} on $iface"

        if [[ "$PLATFORM" == "macos" ]]; then
            sudo ifconfig "$iface" alias "$HOST_ETH_IP" netmask 255.255.255.0
        else
            sudo ip addr add "$HOST_ETH_IP/$NETMASK" dev "$iface" 2>/dev/null || true
            sudo ip link set "$iface" up
        fi

        sleep 0.5
        if [[ "$PLATFORM" == "macos" ]]; then
            if ifconfig "$iface" 2>/dev/null | grep -q "$HOST_ETH_IP"; then
                ok "IP address assigned successfully"
            else
                err "Failed to assign IP. Try manually:"
                echo "  sudo ifconfig $iface alias $HOST_ETH_IP netmask 255.255.255.0"
                exit 1
            fi
        else
            if ip addr show "$iface" 2>/dev/null | grep -q "$HOST_ETH_IP"; then
                ok "IP address assigned successfully"
            else
                err "Failed to assign IP. Try manually:"
                echo "  sudo ip addr add $HOST_ETH_IP/$NETMASK dev $iface"
                exit 1
            fi
        fi
    fi
}

# Set up NAT so the robot can reach the internet through this machine.
setup_nat() {
    local eth_iface="$1"

    header "Internet sharing (NAT forwarding)"

    if [[ "$PLATFORM" == "macos" ]]; then
        local WIFI_IFACE=""
        while IFS= read -r line; do
            if [[ "$line" == "Hardware Port: Wi-Fi" ]]; then
                IFS= read -r next_line
                if [[ "$next_line" == "Device: "* ]]; then
                    WIFI_IFACE="${next_line#Device: }"
                fi
            fi
        done < <(networksetup -listallhardwareports 2>/dev/null)

        if [[ -z "$WIFI_IFACE" ]]; then
            warn "Could not detect Wi-Fi interface — skipping NAT setup"
            return
        fi

        info "Enabling IP forwarding on macOS..."
        sudo sysctl -w net.inet.ip.forwarding=1 >/dev/null 2>&1

        info "Setting up NAT: ${ETH_SUBNET}.0/24 -> $WIFI_IFACE (Wi-Fi)"
        local NAT_CONF="/tmp/unitree_nat.conf"
        echo "nat on $WIFI_IFACE from ${ETH_SUBNET}.0/24 to any -> ($WIFI_IFACE)" > "$NAT_CONF"

        if sudo pfctl -ef "$NAT_CONF" 2>&1 | grep -v "^No ALTQ"; then
            true
        fi

        if sudo pfctl -s nat 2>/dev/null | grep -q "nat on"; then
            ok "NAT enabled: robot traffic forwarded through $WIFI_IFACE"
        else
            err "Failed to load NAT rule. Try manually:"
            echo "  echo 'nat on $WIFI_IFACE from ${ETH_SUBNET}.0/24 to any -> ($WIFI_IFACE)' | sudo pfctl -ef -"
        fi
    elif [[ "$PLATFORM" == "linux" ]]; then
        info "Enabling IP forwarding on Linux..."
        sudo sysctl -w net.ipv4.ip_forward=1 >/dev/null 2>&1

        info "Setting up NAT via iptables..."
        sudo iptables -t nat -A POSTROUTING -s "${ETH_SUBNET}.0/24" ! -o "$eth_iface" -j MASQUERADE 2>/dev/null

        ok "NAT enabled: robot traffic forwarded to default route"
    fi
}

# Configure the robot's default gateway and DNS via SSH over Ethernet.
configure_robot_gateway() {
    header "Configuring robot internet access (via SSH)"

    info "Setting default gateway and DNS on robot ($ROBOT_PC_ETH)..."
    info "Enter robot password when prompted (default: 123)"
    if ssh -t -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$ROBOT_USER@$ROBOT_PC_ETH" \
        "sudo ip route replace default via $HOST_ETH_IP && echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf >/dev/null"
    then
        ok "Robot gateway set to $HOST_ETH_IP, DNS set to 8.8.8.8"

        info "Verifying robot internet access..."
        if ssh -o ConnectTimeout=5 "$ROBOT_USER@$ROBOT_PC_ETH" "ping -c 1 -W 3 8.8.8.8" 2>/dev/null; then
            ok "Robot can reach the internet"
        else
            warn "Robot cannot reach 8.8.8.8 — NAT may not be working"
            echo "  Check that your Mac has an active internet connection (Wi-Fi)"
        fi
    else
        warn "Could not SSH into robot at $ROBOT_PC_ETH"
        echo "  You may need to configure the robot manually:"
        echo "    ssh $ROBOT_USER@$ROBOT_PC_ETH"
        echo "    sudo ip route replace default via $HOST_ETH_IP"
        echo "    echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf"
    fi
}

# ══════════════════════════════════════════════════════════════════════
# ETHERNET MODE
# ══════════════════════════════════════════════════════════════════════

setup_ethernet() {
    local iface_arg="${1:-}"

    detect_ethernet_interface "$iface_arg"
    remove_stale_ips "$IFACE"
    assign_static_ip "$IFACE"

    # Ping robot
    header "Testing connectivity to robot ($ROBOT_MOTOR_IP)"
    info "Pinging $ROBOT_MOTOR_IP ..."
    if ping -c "$PING_COUNT" -W "$PING_TIMEOUT" "$ROBOT_MOTOR_IP" &>/dev/null; then
        ok "Robot is reachable at $ROBOT_MOTOR_IP"
    else
        warn "Cannot ping $ROBOT_MOTOR_IP"
        echo ""
        echo "  Troubleshooting:"
        echo "    - Is the robot powered on and fully booted? (wait ~60s after power)"
        echo "    - Is the Ethernet cable connected to the robot's Ethernet port?"
        echo "    - Try: ping $ROBOT_MOTOR_IP"
        echo ""
        echo "  DDS may still work even if ping fails (some firmware blocks ICMP)."
    fi

    setup_nat "$IFACE"
    configure_robot_gateway

    # Summary
    header "Ethernet Network Configuration"
    echo ""
    echo "  Interface:  $IFACE"
    echo "  Host IP:    $HOST_ETH_IP"
    echo "  Robot PC:   $ROBOT_PC_ETH"
    echo "  Motor board: $ROBOT_MOTOR_IP"
    echo "  Subnet:     ${ETH_SUBNET}.0/24"
    echo "  DDS domain: 0"

    header "Next Steps"
    echo ""
    echo -e "  ${BOLD}# Deploy code to the robot${NC}"
    echo "  ./scripts/deploy_to_robot.sh"
    echo ""
    echo -e "  ${BOLD}# SSH into the robot and run${NC}"
    echo "  ssh $ROBOT_USER@$ROBOT_PC_ETH"
    echo "  cd ~/unitree_launcher && uv run real --gantry"
    echo ""
    echo -e "  ${BOLD}# Mirror real robot state from this Mac${NC}"
    echo "  uv run mirror --gui --interface $IFACE"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════
# WIFI MODE
# ══════════════════════════════════════════════════════════════════════

# Try to find the robot on the WiFi network via ARP or ping scan.
find_robot_on_wifi() {
    # Check ARP table for unitree.lan or known Unitree MAC prefix (fc:23:cd)
    local arp_ip
    arp_ip=$(arp -a 2>/dev/null | grep -i "unitree" | awk -F'[()]' '{print $2}' | head -1)
    if [[ -n "$arp_ip" ]]; then
        echo "$arp_ip"
        return
    fi

    # Check for Unitree MAC prefix in ARP
    arp_ip=$(arp -a 2>/dev/null | grep -i "fc:23:cd" | awk -F'[()]' '{print $2}' | head -1)
    if [[ -n "$arp_ip" ]]; then
        echo "$arp_ip"
        return
    fi
}

# Verify SSH connectivity to a given IP.
verify_wifi_ssh() {
    local ip="$1"
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
        "$ROBOT_USER@$ip" "echo ok" &>/dev/null
}

setup_wifi() {
    local wifi_ssid="${1:-}"
    local wifi_pass="${2:-}"

    header "WiFi Network Setup"

    # ── Get WiFi credentials ──────────────────────────────────────────
    if [[ -z "$wifi_ssid" ]]; then
        echo ""
        read -rp "  WiFi SSID: " wifi_ssid
        if [[ -z "$wifi_ssid" ]]; then
            err "SSID cannot be empty"
            exit 1
        fi
    fi
    if [[ -z "$wifi_pass" ]]; then
        read -rsp "  WiFi password: " wifi_pass
        echo ""
        if [[ -z "$wifi_pass" ]]; then
            err "Password cannot be empty"
            exit 1
        fi
    fi

    # ── Check if robot is already on WiFi ─────────────────────────────
    header "Checking if robot is already on WiFi"

    local robot_wifi_ip=""

    # Try to find via SSH to known Ethernet IP asking for wlan0 address
    # (works if Ethernet is also connected)
    if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o BatchMode=yes \
        "$ROBOT_USER@$ROBOT_PC_ETH" "echo ok" &>/dev/null 2>&1; then
        robot_wifi_ip=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o BatchMode=yes \
            "$ROBOT_USER@$ROBOT_PC_ETH" \
            "ip addr show wlan0 2>/dev/null | awk '/inet /{print \$2}' | cut -d/ -f1" 2>/dev/null || true)
    fi

    # If we got a WiFi IP, verify it's reachable and SSH works
    if [[ -n "$robot_wifi_ip" ]]; then
        if verify_wifi_ssh "$robot_wifi_ip"; then
            ok "Robot already on WiFi at $robot_wifi_ip (SSH key auth works)"
            print_wifi_summary "$robot_wifi_ip"
            return 0
        elif ping -c 1 -W 2 "$robot_wifi_ip" &>/dev/null; then
            ok "Robot already on WiFi at $robot_wifi_ip"
            info "Setting up SSH key auth..."
            setup_ssh_keys "$robot_wifi_ip"
            print_wifi_summary "$robot_wifi_ip"
            return 0
        fi
    fi

    # Try ARP/network discovery
    info "Scanning local network for robot..."
    local discovered_ip
    discovered_ip=$(find_robot_on_wifi)
    if [[ -n "$discovered_ip" ]]; then
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o PasswordAuthentication=no \
            "$ROBOT_USER@$discovered_ip" "echo ok" &>/dev/null; then
            ok "Found robot at $discovered_ip (SSH works)"
            robot_wifi_ip="$discovered_ip"
            print_wifi_summary "$robot_wifi_ip"
            return 0
        fi
        # Found device but can't SSH — might be the internal network module, not the PC
        info "Found Unitree device at $discovered_ip but SSH not available there"
    fi

    info "Robot not found on WiFi — bootstrapping via Ethernet"

    # ── Bootstrap: set up Ethernet connection ─────────────────────────
    header "Setting up Ethernet for bootstrap"

    local iface=""
    if [[ "$PLATFORM" == "macos" ]]; then
        ensure_macos_network_service
        iface=$(find_macos_ethernet)
    else
        iface=$(find_linux_ethernet)
    fi

    if [[ -z "$iface" ]]; then
        err "No active Ethernet interface found."
        echo ""
        echo "  To bootstrap WiFi, you need a one-time Ethernet connection."
        echo "  Plug in an Ethernet cable between your computer and the robot,"
        echo "  then re-run:  $0 wifi"
        echo ""
        if [[ "$PLATFORM" == "macos" ]]; then
            echo "  Available wired ports:"
            list_macos_wired_ports
        fi
        exit 1
    fi

    IFACE="$iface"
    info "Using Ethernet interface: ${BOLD}$IFACE${NC}"

    # Assign static IP if needed (minimal version — skip NAT/gateway for bootstrap)
    local current_ips
    if [[ "$PLATFORM" == "macos" ]]; then
        current_ips=$(ifconfig "$IFACE" 2>/dev/null | awk '/inet /{print $2}')
    else
        current_ips=$(ip -4 addr show "$IFACE" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1)
    fi

    if ! echo "$current_ips" | grep -q "^${HOST_ETH_IP}$"; then
        info "Assigning ${HOST_ETH_IP} on $IFACE ..."
        if [[ "$PLATFORM" == "macos" ]]; then
            sudo ifconfig "$IFACE" alias "$HOST_ETH_IP" netmask 255.255.255.0
        else
            sudo ip addr add "$HOST_ETH_IP/$NETMASK" dev "$IFACE" 2>/dev/null || true
            sudo ip link set "$IFACE" up
        fi
        sleep 0.5
    fi
    ok "Ethernet IP: $HOST_ETH_IP on $IFACE"

    # Verify Ethernet SSH
    header "Connecting to robot via Ethernet"
    info "Pinging robot PC at $ROBOT_PC_ETH ..."
    if ! ping -c 1 -W "$PING_TIMEOUT" "$ROBOT_PC_ETH" &>/dev/null; then
        err "Cannot reach robot PC at $ROBOT_PC_ETH"
        echo "  - Is the robot powered on and fully booted?"
        echo "  - Is the Ethernet cable connected?"
        exit 1
    fi
    ok "Robot PC reachable at $ROBOT_PC_ETH"

    # Set up SSH keys over Ethernet first (so later commands don't need passwords)
    setup_ssh_keys "$ROBOT_PC_ETH"

    # ── Enable WiFi on the robot ──────────────────────────────────────
    header "Enabling WiFi on robot"

    # SSH ControlMaster: open one connection, reuse for all commands (one password prompt)
    SSH_CTRL="/tmp/unitree-wifi-$$"
    cleanup_ssh_ctrl() {
        ssh -o ControlPath="$SSH_CTRL" -O exit "$ROBOT_USER@$ROBOT_PC_ETH" 2>/dev/null || true
        rm -f "$SSH_CTRL" 2>/dev/null || true
    }
    trap cleanup_ssh_ctrl EXIT

    info "Opening persistent SSH session (enter password once if prompted)..."
    if ! ssh -o ControlMaster=yes -o ControlPath="$SSH_CTRL" -o ControlPersist=300 \
        -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "$ROBOT_USER@$ROBOT_PC_ETH" true; then
        err "Cannot establish SSH session to $ROBOT_PC_ETH"
        exit 1
    fi
    ok "SSH session established (all commands will reuse this connection)"

    # Run all WiFi setup in a single interactive session (one sudo prompt).
    # Output goes directly to terminal so the user sees sudo prompts and progress.
    info "Enabling WiFi radio and connecting to '${wifi_ssid}' ..."
    info "Enter sudo password on robot when prompted (default: 123)"
    echo ""

    local wifi_ok=true
    ssh -t -o ControlPath="$SSH_CTRL" -o StrictHostKeyChecking=no \
        "$ROBOT_USER@$ROBOT_PC_ETH" "\
set -e
sudo -v
echo '  WiFi radio: unblocking...'
sudo rfkill unblock wifi
echo '  WiFi radio: unblocked'
echo '  NetworkManager: enabling WiFi...'
sudo nmcli radio wifi on
echo '  NetworkManager: WiFi enabled'
echo '  Waiting for wlan0 to become ready...'
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    state=\$(nmcli -t -f DEVICE,STATE device status 2>/dev/null | grep '^wlan0:' || true)
    case \"\$state\" in *disconnected*|*connected*) break ;; esac
    sleep 1
done
echo '  Connecting to WiFi network...'
sudo nmcli device wifi connect '$wifi_ssid' password '$wifi_pass'" \
    || wifi_ok=false

    echo ""
    if [[ "$wifi_ok" == true ]]; then
        ok "Robot connected to '${wifi_ssid}'"
    else
        err "Failed to connect robot to WiFi"
        echo ""
        echo "  Troubleshooting:"
        echo "    - Is the SSID and password correct?"
        echo "    - Is the WiFi network in range?"
        echo "    SSH in manually and try:"
        echo "      ssh $ROBOT_USER@$ROBOT_PC_ETH"
        echo "      sudo nmcli device wifi list"
        echo "      sudo nmcli device wifi connect \"$wifi_ssid\" password \"<password>\""
        exit 1
    fi

    # ── Get robot WiFi IP ─────────────────────────────────────────────
    info "Getting robot WiFi IP..."
    sleep 2  # wait for DHCP

    robot_wifi_ip=$(ssh -o ControlPath="$SSH_CTRL" -o StrictHostKeyChecking=no \
        "$ROBOT_USER@$ROBOT_PC_ETH" \
        "ip addr show wlan0 | awk '/inet /{print \$2}' | cut -d/ -f1" 2>/dev/null)

    if [[ -z "$robot_wifi_ip" ]]; then
        err "Robot connected to WiFi but didn't get an IP address"
        echo "  Check DHCP on your WiFi router."
        exit 1
    fi
    ok "Robot WiFi IP: ${BOLD}$robot_wifi_ip${NC}"

    # ── Verify SSH over WiFi ──────────────────────────────────────────
    header "Verifying SSH over WiFi"

    info "Pinging robot at $robot_wifi_ip ..."
    if ! ping -c 2 -W "$PING_TIMEOUT" "$robot_wifi_ip" &>/dev/null; then
        err "Cannot ping robot at $robot_wifi_ip"
        echo "  Both machines are on the WiFi network but can't communicate."
        echo "  Check that your WiFi router allows client-to-client traffic."
        exit 1
    fi
    ok "Robot is reachable at $robot_wifi_ip"

    # Copy SSH keys to the WiFi IP as well
    setup_ssh_keys "$robot_wifi_ip"

    # Verify full SSH connectivity
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
        "$ROBOT_USER@$robot_wifi_ip" "echo ok" &>/dev/null; then
        ok "SSH over WiFi works (key auth)"
    else
        info "Testing SSH with password..."
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
            "$ROBOT_USER@$robot_wifi_ip" "echo ok" </dev/null 2>/dev/null; then
            ok "SSH over WiFi works (password auth)"
        else
            err "Cannot SSH to robot at $robot_wifi_ip"
            echo "  Ping works but SSH doesn't. Check the robot's SSH config."
            exit 1
        fi
    fi

    # ── Make WiFi persist across reboots ──────────────────────────────
    header "Persisting WiFi across reboots"

    # NetworkManager saves the connection profile automatically, but rfkill
    # soft-block may return on reboot. Create a systemd service to unblock it.
    local persist_result
    persist_result=$(ssh -o ControlPath="$SSH_CTRL" -o StrictHostKeyChecking=no \
        "$ROBOT_USER@$ROBOT_PC_ETH" bash -s 2>/dev/null <<'PERSIST_SCRIPT'
        # Check if the persist service already exists
        if systemctl is-enabled unitree-wifi.service &>/dev/null; then
            echo "already_exists"
            exit 0
        fi

        # Create systemd service to unblock WiFi and enable NM WiFi on boot
        sudo tee /etc/systemd/system/unitree-wifi.service >/dev/null <<'UNIT'
[Unit]
Description=Unblock WiFi radio for Unitree G1
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=oneshot
ExecStart=/usr/sbin/rfkill unblock wifi
ExecStartPost=/usr/bin/nmcli radio wifi on
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT

        sudo systemctl daemon-reload
        sudo systemctl enable unitree-wifi.service
        echo "created"
PERSIST_SCRIPT
    ) || persist_result="failed"

    case "$persist_result" in
        *already_exists*) ok "WiFi persistence service already installed" ;;
        *created*)        ok "WiFi will auto-enable on robot boot" ;;
        *)                warn "Could not install WiFi persistence service — WiFi may not survive reboot"
                          echo "  You can re-run this script after a reboot to re-enable WiFi." ;;
    esac

    print_wifi_summary "$robot_wifi_ip"
}

print_wifi_summary() {
    local robot_ip="$1"

    header "WiFi Network Configuration"
    echo ""
    echo "  Robot WiFi IP: $robot_ip"
    echo "  Robot user:    $ROBOT_USER"
    echo "  SSH:           ssh $ROBOT_USER@$robot_ip"
    echo ""
    echo "  The Ethernet cable can now be disconnected."
    echo "  WiFi will reconnect automatically after robot reboots."

    header "Next Steps"
    echo ""
    echo -e "  ${BOLD}# Deploy code to the robot${NC}"
    echo "  ./scripts/deploy_to_robot.sh $robot_ip"
    echo ""
    echo -e "  ${BOLD}# SSH into the robot and run${NC}"
    echo "  ssh $ROBOT_USER@$robot_ip"
    echo "  cd ~/unitree_launcher && uv run real --gantry"
    echo ""
    echo -e "  ${BOLD}# Get logs from robot${NC}"
    echo "  ./scripts/get_logs_from_robot.sh $robot_ip"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════
# MODE DISPATCH
# ══════════════════════════════════════════════════════════════════════

MODE="${1:-}"
shift || true

case "$MODE" in
    ethernet) setup_ethernet "$@" ;;
    wifi)     setup_wifi "$@" ;;
    --help|-h) ;; # handled above
    "")
        err "Please specify a mode: ethernet or wifi"
        echo ""
        echo "  Usage:"
        echo "    $0 ethernet [INTERFACE]"
        echo "    $0 wifi [SSID] [PASSWORD]"
        echo "    $0 --help"
        exit 1
        ;;
    *)
        err "Unknown mode: $MODE"
        echo ""
        echo "  Usage:"
        echo "    $0 ethernet [INTERFACE]"
        echo "    $0 wifi [SSID] [PASSWORD]"
        echo "    $0 --help"
        exit 1
        ;;
esac
