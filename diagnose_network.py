import socket
import subprocess
import sys
import time
import requests


def get_local_ip():
    """Findet die lokale IPv4-Adresse."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unbekannt"


def check_port(port=5000):
    """Prüft, ob Port 5000 offen ist."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(("0.0.0.0", port))
    sock.close()
    return result == 0


def check_firewall():
    """Prüft, ob Windows-Firewall Port 5000 blockiert."""
    try:
        output = subprocess.check_output(
            'netsh advfirewall firewall show rule name="Flask5000"',
            shell=True,
            text=True,
        )
        if "No rules match" in output or "Keine Regeln" in output:
            return False
        return True
    except:
        return False


def check_flask_running(ip):
    """Prüft, ob Flask erreichbar ist."""
    try:
        url = f"http://{ip}:5000/"
        r = requests.get(url, timeout=1)
        return True
    except:
        return False


print("🔍 Netzwerk-Diagnose startet...\n")
time.sleep(1)

# 1. IP anzeigen
ip = get_local_ip()
print(f"📌 Deine lokale IP-Adresse: {ip}")

# 2. Port prüfen
print("\n🔌 Prüfe, ob Port 5000 offen ist...")
if check_port(5000):
    print("✅ Port 5000 ist offen")
else:
    print("❌ Port 5000 ist GESCHLOSSEN (Flask läuft nicht oder Firewall blockiert)")

# 3. Firewall prüfen
print("\n🛡 Prüfe Windows-Firewall-Regel...")
if check_firewall():
    print("✅ Firewall-Regel 'Flask5000' existiert")
else:
    print("❌ Keine Firewall-Regel für Port 5000 gefunden")
    print("   → Führe in PowerShell (Admin) aus:")
    print(
        '     netsh advfirewall firewall add rule name="Flask5000" dir=in action=allow protocol=TCP localport=5000'
    )

# 4. Flask erreichbar?
print("\n🌐 Prüfe, ob Flask erreichbar ist...")
if check_flask_running(ip):
    print(f"✅ Flask ist erreichbar unter: http://{ip}:5000/")
else:
    print(f"❌ Flask NICHT erreichbar unter: http://{ip}:5000/")
    print("   → Läuft dein Flask-Server?")
    print("   → Hast du host='0.0.0.0' gesetzt?")

print("\n📱 Wenn alles grün ist, öffne auf deinem Handy:")
print(f"👉 http://{ip}:5000/scan.jpg")
print(f"👉 oder http://{ip}:5000/")
