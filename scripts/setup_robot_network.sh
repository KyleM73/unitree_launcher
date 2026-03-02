#!/bin/bash
# setup_robot_network.sh — Configure Ethernet for Unitree G1 communication.
#
# Detects the Ethernet interface, assigns a static IP on the robot's subnet,
# verifies connectivity, and prints example commands.
#
# Usage:
#   ./scripts/setup_robot_network.sh              # auto-detect interface
#   ./scripts/setup_robot_network.sh en8           # specify interface
#   ./scripts/setup_robot_network.sh --help

set -euo pipefail

# ── Constants ──────────────────────────────────────────────────────────
ROBOT_IP="192.168.123.161"
SUBNET="192.168.123"
HOST_IP="${SUBNET}.100"
NETMASK="24"
PING_TIMEOUT=2
PING_COUNT=3

# ── Colors ─────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()   { echo -e "${RED}[error]${NC} $*"; }
header(){ echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Help ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Setup Ethernet for Unitree G1 robot communication.

Usage:
    ./scripts/setup_robot_network.sh [INTERFACE]

Arguments:
    INTERFACE   Ethernet interface name (e.g. en8, enp3s0).
                If omitted, auto-detects from connected Ethernet adapters.

What it does:
    1. Detects (or uses specified) Ethernet interface
    2. On macOS: creates a network service if missing (required to activate the port)
    3. Removes stale robot-subnet IPs from other interfaces (fixes routing conflicts)
    4. Assigns static IP 192.168.123.100/24
    5. Pings the robot at 192.168.123.161
    6. Prints example commands to run (including Docker on Linux)

Network layout:
    Host:  192.168.123.100
    Robot: 192.168.123.161
    DDS domain: 0

Note:
    This script runs on the HOST machine (not inside Docker).
    Docker containers use --network host to share the host's network stack,
    so run this script first, then use docker compose.
EOF
    exit 0
fi

# ── Detect OS ──────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)      err "Unsupported OS: $OS"; exit 1 ;;
esac

# ── Find Ethernet interface ────────────────────────────────────────────
header "Detecting Ethernet interface"

get_macos_wired_ports() {
    # Returns lines of "device_name\thardware_port_name" for all wired ports.
    # On macOS, wired Ethernet can show up as:
    #   "Ethernet Adapter (en4)"      — built-in Thunderbolt ports
    #   "USB 10/100/1000 LAN"         — USB-C Ethernet dongles
    #   "Thunderbolt Ethernet Slot 0" — Thunderbolt docks
    #   "Belkin USB-C LAN"            — third-party adapters
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
    # macOS deactivates Ethernet interfaces that have no network service
    # (e.g. if the user deleted the profile in System Settings).
    # Check if any wired port is inactive due to a missing service,
    # and if so, create one.

    # First, check if there's already an active wired interface — if so, skip.
    while IFS=$'\t' read -r dev port_name; do
        if ifconfig "$dev" 2>/dev/null | grep -q "status: active"; then
            return  # Already have an active wired port, nothing to do
        fi
    done < <(get_macos_wired_ports)

    # No active wired interface. Check if any are missing a network service.
    local existing_services
    existing_services=$(networksetup -listallnetworkservices 2>/dev/null)
    local needs_service=""
    local needs_service_port=""

    while IFS=$'\t' read -r dev port_name; do
        # Check if this hardware port has a corresponding network service.
        # Services can be named anything, so check by hardware port name
        # and also check for our custom "Unitree G1" name.
        local has_service=false
        if echo "$existing_services" | grep -qF "$port_name"; then
            has_service=true
        fi
        if echo "$existing_services" | grep -qF "Unitree G1"; then
            has_service=true
        fi
        # Also check: does -getinfo work for the port name?
        if networksetup -getinfo "$port_name" &>/dev/null; then
            has_service=true
        fi

        if [[ "$has_service" == false && -z "$needs_service" ]]; then
            needs_service="$dev"
            needs_service_port="$port_name"
        fi
    done < <(get_macos_wired_ports)

    if [[ -z "$needs_service" ]]; then
        return  # All ports have services (they're just inactive = no cable)
    fi

    warn "No network service for '$needs_service_port' ($needs_service)"
    warn "macOS won't activate the interface without a network profile."
    info "Creating service 'Unitree G1' on $needs_service with static IP $HOST_IP ..."
    echo ""
    if sudo networksetup -createnetworkservice "Unitree G1" "$needs_service" && \
       sudo networksetup -setmanual "Unitree G1" "$HOST_IP" "255.255.255.0"; then
        ok "Created service 'Unitree G1' on $needs_service"
        info "Waiting for interface to activate ..."
        sleep 2
    else
        err "Could not create network service."
        echo ""
        echo "  Run these manually:"
        echo "    sudo networksetup -createnetworkservice \"Unitree G1\" $needs_service"
        echo "    sudo networksetup -setmanual \"Unitree G1\" $HOST_IP 255.255.255.0"
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

    # Prefer the interface that is status: active (cable plugged in)
    for iface in $all_wired_ifaces; do
        if ifconfig "$iface" 2>/dev/null | grep -q "status: active"; then
            echo "$iface"
            return
        fi
    done

    # No active interface found
    return
}

find_linux_ethernet() {
    # Find physical Ethernet devices with carrier (cable plugged in)
    local ifaces
    ifaces=$(ip -o link show 2>/dev/null \
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
            fi
        done)

    echo "$ifaces" | head -1
}

list_macos_wired_ports() {
    # Show all wired-ish hardware ports with their status for diagnostics
    local current_port=""
    while IFS= read -r line; do
        if [[ "$line" == "Hardware Port: "* ]]; then
            current_port="${line#Hardware Port: }"
        elif [[ "$line" == "Device: "* && -n "$current_port" ]]; then
            local dev="${line#Device: }"
            case "$current_port" in
                *Wi-Fi*|*AirPort*|*Bluetooth*|*FireWire*|*VPN*|*VLAN*)
                    ;;
                *)
                    local status
                    status=$(ifconfig "$dev" 2>/dev/null | awk '/status:/{print $2}')
                    printf "    %-8s  %-35s  %s\n" "$dev" "$current_port" "${status:-unknown}"
                    ;;
            esac
            current_port=""
        fi
    done < <(networksetup -listallhardwareports 2>/dev/null)
}

if [[ -n "${1:-}" ]]; then
    IFACE="$1"
    info "Using specified interface: $IFACE"
else
    # On macOS, ensure network services exist (otherwise interfaces stay inactive)
    if [[ "$PLATFORM" == "macos" ]]; then
        ensure_macos_network_service
        IFACE=$(find_macos_ethernet)
    else
        IFACE=$(find_linux_ethernet)
    fi

    if [[ -z "$IFACE" ]]; then
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
        echo "    $0 <interface_name>"
        exit 1
    fi
    info "Auto-detected Ethernet interface: ${BOLD}$IFACE${NC}"
fi

# Verify interface exists
if [[ "$PLATFORM" == "macos" ]]; then
    if ! ifconfig "$IFACE" &>/dev/null; then
        err "Interface '$IFACE' does not exist."
        echo "  Available wired ports:"
        list_macos_wired_ports
        exit 1
    fi
else
    if ! ip link show "$IFACE" &>/dev/null; then
        err "Interface '$IFACE' does not exist."
        echo "  Available interfaces:"
        ip -o link show | awk -F': ' '{print "    " $2}'
        exit 1
    fi
fi

ok "Interface $IFACE exists"

# Show link info
if [[ "$PLATFORM" == "macos" ]]; then
    link_status=$(ifconfig "$IFACE" 2>/dev/null | awk '/status:/{print $2}')
    media_info=$(ifconfig "$IFACE" 2>/dev/null | awk '/media:/{$1=""; print}' | xargs)
    if [[ "$link_status" == "active" ]]; then
        ok "Link is active ($media_info)"
    else
        warn "Link status: $link_status"
        warn "Make sure the Ethernet cable is plugged in to this port."
    fi
else
    carrier=$(cat "/sys/class/net/$IFACE/carrier" 2>/dev/null || echo "0")
    speed=$(cat "/sys/class/net/$IFACE/speed" 2>/dev/null || echo "?")
    if [[ "$carrier" == "1" ]]; then
        ok "Link is active (${speed} Mbps)"
    else
        warn "No carrier detected on $IFACE"
        warn "Make sure the Ethernet cable is plugged in."
    fi
fi

# ── Remove stale IPs from other interfaces ─────────────────────────────
# If another interface has an address on 192.168.123.x, macOS/Linux will
# route robot traffic there instead of $IFACE. This happens when you
# previously configured a different port and then moved the cable.
header "Checking for conflicting IPs on other interfaces"

if [[ "$PLATFORM" == "macos" ]]; then
    stale_found=false
    for other_iface in $(ifconfig -l); do
        [[ "$other_iface" == "$IFACE" ]] && continue
        other_ips=$(ifconfig "$other_iface" 2>/dev/null | awk '/inet /{print $2}')
        while IFS= read -r ip; do
            if [[ "$ip" == ${SUBNET}.* ]]; then
                warn "Found ${SUBNET}.x address $ip on $other_iface (not our target interface)"
                info "Removing $ip from $other_iface to fix routing ..."
                sudo ifconfig "$other_iface" -alias "$ip" 2>/dev/null && \
                    ok "Removed $ip from $other_iface" || \
                    warn "Could not remove $ip from $other_iface (may need manual removal)"
                stale_found=true
            fi
        done <<< "$other_ips"
    done
    if [[ "$stale_found" == false ]]; then
        ok "No conflicting IPs found"
    fi
else
    stale_found=false
    for other_iface in $(ip -o link show 2>/dev/null | awk -F': ' '{print $2}'); do
        [[ "$other_iface" == "$IFACE" ]] && continue
        other_ips=$(ip -4 addr show "$other_iface" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1)
        while IFS= read -r ip; do
            if [[ "$ip" == ${SUBNET}.* ]]; then
                warn "Found ${SUBNET}.x address $ip on $other_iface (not our target interface)"
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

# ── Check if already configured ────────────────────────────────────────
header "Checking current IP configuration"

already_configured=false
if [[ "$PLATFORM" == "macos" ]]; then
    current_ips=$(ifconfig "$IFACE" 2>/dev/null | awk '/inet /{print $2}')
else
    current_ips=$(ip -4 addr show "$IFACE" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1)
fi

if echo "$current_ips" | grep -q "^${HOST_IP}$"; then
    ok "Interface already has $HOST_IP"
    already_configured=true
elif [[ -n "$current_ips" ]]; then
    info "Current IPs on $IFACE:"
    echo "$current_ips" | sed 's/^/    /'
fi

# ── Assign static IP ──────────────────────────────────────────────────
if [[ "$already_configured" == false ]]; then
    header "Assigning static IP"
    info "Setting ${BOLD}$HOST_IP/$NETMASK${NC} on $IFACE"

    if [[ "$PLATFORM" == "macos" ]]; then
        sudo ifconfig "$IFACE" alias "$HOST_IP" netmask 255.255.255.0
    else
        sudo ip addr add "$HOST_IP/$NETMASK" dev "$IFACE" 2>/dev/null || true
        sudo ip link set "$IFACE" up
    fi

    # Verify it was set
    sleep 0.5
    if [[ "$PLATFORM" == "macos" ]]; then
        if ifconfig "$IFACE" 2>/dev/null | grep -q "$HOST_IP"; then
            ok "IP address assigned successfully"
        else
            err "Failed to assign IP. Try manually:"
            echo "  sudo ifconfig $IFACE alias $HOST_IP netmask 255.255.255.0"
            exit 1
        fi
    else
        if ip addr show "$IFACE" 2>/dev/null | grep -q "$HOST_IP"; then
            ok "IP address assigned successfully"
        else
            err "Failed to assign IP. Try manually:"
            echo "  sudo ip addr add $HOST_IP/$NETMASK dev $IFACE"
            exit 1
        fi
    fi
fi

# ── Ping robot ─────────────────────────────────────────────────────────
header "Testing connectivity to robot ($ROBOT_IP)"

info "Pinging $ROBOT_IP ..."
if ping -c "$PING_COUNT" -W "$PING_TIMEOUT" "$ROBOT_IP" &>/dev/null; then
    ok "Robot is reachable at $ROBOT_IP"
else
    warn "Cannot ping $ROBOT_IP"
    echo ""
    echo "  Troubleshooting:"
    echo "    - Is the robot powered on and fully booted? (wait ~60s after power)"
    echo "    - Is the Ethernet cable connected to the robot's Ethernet port?"
    echo "    - Try: ping $ROBOT_IP"
    echo ""
    echo "  DDS may still work even if ping fails (some firmware blocks ICMP)."
    echo "  Try the DDS verification command below to check."
fi

# ── Summary ────────────────────────────────────────────────────────────
header "Network Configuration Summary"
echo ""
echo "  Interface:  $IFACE"
echo "  Host IP:    $HOST_IP"
echo "  Robot IP:   $ROBOT_IP"
echo "  Subnet:     ${SUBNET}.0/24"
echo "  DDS domain: 0"
echo ""

# ── Example commands ───────────────────────────────────────────────────
header "Example Commands"
echo ""
echo -e "  ${BOLD}# Verify DDS communication (should print mode_machine=5)${NC}"
echo "  python scripts/tests/test_dds.py --interface $IFACE"
echo ""
echo -e "  ${BOLD}# Right arm shoulder test (safe low-amplitude sinusoid)${NC}"
echo "  python scripts/tests/test_arm_reach.py real --interface $IFACE"
echo ""
echo -e "  ${BOLD}# Run a policy on the real robot${NC}"
echo "  python -m unitree_launcher.main real --policy assets/policies/stance_29dof.onnx --interface $IFACE"
echo ""
echo -e "  ${BOLD}# Mirror real robot state in MuJoCo viewer${NC}"
echo "  mjpython scripts/mirror_real_robot.py --interface $IFACE"

if [[ "$PLATFORM" == "linux" ]]; then
    echo ""
    echo -e "  ${BOLD}# Docker: run policy on real robot (uses host networking)${NC}"
    echo "  docker compose -f docker/docker-compose.yml --profile real run --rm real-robot \\"
    echo "      real --policy assets/policies/stance_29dof.onnx --interface $IFACE"
fi
echo ""
