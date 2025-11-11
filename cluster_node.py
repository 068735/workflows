#!/usr/bin/env python3
import asyncio
import json
import logging
import psutil
import subprocess
import time
from typing import Dict, List, Optional
import requests
import threading
from dataclasses import dataclass
from cryptography.fernet import Fernet
import hashlib
import base64

@dataclass
class NodeConfig:
    node_id: str
    admin_id: str
    worker_url: str
    encryption_password: str
    easytier_network: str
    monitor_interval: int = 30
    report_interval: int = 60

class EasyTierManager:
    """管理 EasyTier 连接和监控"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.network_secret = self._generate_network_secret()
        self.connected_peers = []
        self.traffic_stats = {"tx_bytes": 0, "rx_bytes": 0, "tx_packets": 0, "rx_packets": 0}
        
    def _generate_network_secret(self) -> str:
        """生成网络密钥"""
        seed = f"{self.config.admin_id}-{self.config.encryption_password}"
        return hashlib.sha256(seed.encode()).hexdigest()[:32]
    
    def start_easytier(self):
        """启动 EasyTier 连接"""
        cmd = [
            "easytier-core", "-d",
            "--network-name", self.config.easytier_network,
            "--network-secret", self.network_secret,
            "-p", "tcp://public.easytier.cn:11010",
            "--hostname", self.config.node_id
        ]
        
        try:
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info("✅ EasyTier 启动成功")
        except Exception as e:
            logging.error(f"❌ EasyTier 启动失败: {e}")
    
    def get_peer_connections(self) -> List[Dict]:
        """获取对等节点连接信息"""
        try:
            # 使用 easytier-cli 获取节点状态
            result = subprocess.run([
                "easytier-cli", "peer", "list", "--json"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                peers = json.loads(result.stdout)
                return [{
                    "node_id": peer.get("hostname", "unknown"),
                    "latency": peer.get("latency", 0),
                    "endpoint": peer.get("endpoint", ""),
                    "is_local": peer.get("is_local", False)
                } for peer in peers]
        except Exception as e:
            logging.error(f"获取节点连接失败: {e}")
        
        return []
    
    def get_traffic_stats(self) -> Dict:
        """获取流量统计"""
        try:
            # 监控虚拟网卡流量
            for interface in psutil.net_io_counters(pernic=True):
                if interface.startswith("tun") or interface.startswith("utun"):
                    stats = psutil.net_io_counters(pernic=True)[interface]
                    self.traffic_stats = {
                        "tx_bytes": stats.bytes_sent,
                        "rx_bytes": stats.bytes_recv,
                        "tx_packets": stats.packets_sent,
                        "rx_packets": stats.packets_recv
                    }
                    break
        except Exception as e:
            logging.error(f"获取流量统计失败: {e}")
        
        return self.traffic_stats

class SecurityManager:
    """安全管理器：黑白名单和异常检测"""
    
    def __init__(self):
        self.whitelist = set()
        self.blacklist = set()
        self.suspicious_ips = set()
        self.connection_threshold = 100  # 异常连接数阈值
        
    def load_blacklist(self, ips: List[str]):
        """加载黑名单IP"""
        self.blacklist.update(ips)
        logging.info(f"🔄 已加载 {len(ips)} 个黑名单IP")
    
    def load_whitelist(self, ips: List[str]):
        """加载白名单IP"""  
        self.whitelist.update(ips)
        logging.info(f"🔄 已加载 {len(ips)} 个白名单IP")
    
    def check_connection(self, src_ip: str, dst_ip: str, protocol: str) -> bool:
        """检查连接是否允许"""
        if src_ip in self.blacklist:
            logging.warning(f"🚫 拦截黑名单IP连接: {src_ip} -> {dst_ip}")
            return False
            
        if self.whitelist and src_ip not in self.whitelist:
            logging.warning(f"⚠️ 拦截非白名单IP连接: {src_ip} -> {dst_ip}")
            return False
            
        return True
    
    def detect_anomalies(self, connections: List[Dict]) -> List[Dict]:
        """检测异常流量模式"""
        anomalies = []
        ip_connections = {}
        
        # 统计每个IP的连接数
        for conn in connections:
            src_ip = conn.get('src_ip')
            ip_connections[src_ip] = ip_connections.get(src_ip, 0) + 1
        
        # 检测异常
        for ip, count in ip_connections.items():
            if count > self.connection_threshold:
                anomaly = {
                    "type": "high_connection_count",
                    "src_ip": ip,
                    "count": count,
                    "threshold": self.connection_threshold,
                    "timestamp": time.time()
                }
                anomalies.append(anomaly)
                self.suspicious_ips.add(ip)
        
        return anomalies

class ClusterNode:
    """集群节点主类"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.easytier = EasyTierManager(config)
        self.security = SecurityManager()
        self.encryption_key = self._derive_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
        # 状态变量
        self.connected_nodes = {}
        self.node_metrics = {}
        self.last_sync = 0
        
        self.setup_logging()
    
    def _derive_encryption_key(self) -> bytes:
        """派生加密密钥"""
        password = self.config.encryption_password.encode()
        salt = self.config.admin_id.encode()
        kdf = hashlib.pbkdf2_hmac('sha256', password, salt, 100000)
        return base64.urlsafe_b64encode(kdf[:32])
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/cluster-node.log'),
                logging.StreamHandler()
            ]
        )
    
    def start(self):
        """启动节点服务"""
        logging.info(f"🚀 启动集群节点: {self.config.node_id}")
        
        # 启动 EasyTier
        self.easytier.start_easytier()
        
        # 加载安全列表
        self.load_security_lists()
        
        # 启动监控循环
        self.start_monitoring()
        
        # 启动状态报告
        self.start_status_reporting()
    
    def load_security_lists(self):
        """从集群加载安全列表"""
        try:
            response = requests.get(
                f"{self.config.worker_url}/api/security/whitelist",
                headers={"X-Node-ID": self.config.node_id},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self.security.load_whitelist(data.get("whitelist", []))
            
            response = requests.get(
                f"{self.config.worker_url}/api/security/blacklist", 
                headers={"X-Node-ID": self.config.node_id},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self.security.load_blacklist(data.get("blacklist", []))
                
        except Exception as e:
            logging.error(f"加载安全列表失败: {e}")
    
    def collect_metrics(self) -> Dict:
        """收集节点指标"""
        # 系统指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 网络指标
        traffic_stats = self.easytier.get_traffic_stats()
        peer_connections = self.easytier.get_peer_connections()
        
        return {
            "node_id": self.config.node_id,
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            },
            "network": {
                "peer_count": len(peer_connections),
                "traffic": traffic_stats,
                "peers": peer_connections
            },
            "security": {
                "whitelist_count": len(self.security.whitelist),
                "blacklist_count": len(self.security.blacklist),
                "suspicious_count": len(self.security.suspicious_ips)
            }
        }
    
    def start_monitoring(self):
        """启动监控循环"""
        def monitor_loop():
            while True:
                try:
                    # 收集指标
                    metrics = self.collect_metrics()
                    
                    # 检测异常
                    anomalies = self.security.detect_anomalies(
                        metrics["network"]["peers"]
                    )
                    
                    if anomalies:
                        self.report_anomalies(anomalies)
                    
                    time.sleep(self.config.monitor_interval)
                    
                except Exception as e:
                    logging.error(f"监控循环错误: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
    
    def start_status_reporting(self):
        """启动状态报告"""
        def report_loop():
            while True:
                try:
                    metrics = self.collect_metrics()
                    
                    # 加密指标数据
                    encrypted_metrics = self.fernet.encrypt(
                        json.dumps(metrics).encode()
                    )
                    
                    # 报告到集群
                    requests.post(
                        f"{self.config.worker_url}/api/nodes/metrics",
                        headers={
                            "X-Node-ID": self.config.node_id,
                            "Content-Type": "application/octet-stream"
                        },
                        data=encrypted_metrics,
                        timeout=15
                    )
                    
                    time.sleep(self.config.report_interval)
                    
                except Exception as e:
                    logging.error(f"状态报告错误: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=report_loop, daemon=True)
        thread.start()
    
    def report_anomalies(self, anomalies: List[Dict]):
        """报告异常事件"""
        try:
            encrypted_data = self.fernet.encrypt(
                json.dumps(anomalies).encode()
            )
            
            requests.post(
                f"{self.config.worker_url}/api/security/anomalies",
                headers={
                    "X-Node-ID": self.config.node_id,
                    "Content-Type": "application/octet-stream"
                },
                data=encrypted_data,
                timeout=10
            )
            
            logging.warning(f"📢 报告 {len(anomalies)} 个异常事件")
            
        except Exception as e:
            logging.error(f"异常报告失败: {e}")

class NodeCLI:
    """节点命令行交互界面"""
    
    def __init__(self, node: ClusterNode):
        self.node = node
    
    def run(self):
        """运行交互式CLI"""
        while True:
            try:
                print("\n" + "="*50)
                print("🏢 集群节点管理系统")
                print("="*50)
                print("1. 📊 查看节点状态")
                print("2. 🌐 查看网络连接")
                print("3. 🛡️  安全管理")
                print("4. ⚙️  手动同步集群")
                print("5. 🚪 退出")
                
                choice = input("请选择操作 (1-5): ").strip()
                
                if choice == "1":
                    self.show_node_status()
                elif choice == "2":
                    self.show_network_status()
                elif choice == "3":
                    self.security_management()
                elif choice == "4":
                    self.manual_sync()
                elif choice == "5":
                    break
                else:
                    print("❌ 无效选择")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    def show_node_status(self):
        """显示节点状态"""
        metrics = self.node.collect_metrics()
        
        print(f"\n📊 节点 {metrics['node_id']} 状态:")
        print(f"   💻 CPU使用率: {metrics['system']['cpu_percent']}%")
        print(f"   🧠 内存使用率: {metrics['system']['memory_percent']}%")
        print(f"   💾 磁盘使用率: {metrics['system']['disk_percent']}%")
        print(f"   🔗 对等节点数: {metrics['network']['peer_count']}")
        print(f"   📨 发送流量: {metrics['network']['traffic']['tx_bytes']} bytes")
        print(f"   📥 接收流量: {metrics['network']['traffic']['rx_bytes']} bytes")
    
    def show_network_status(self):
        """显示网络状态"""
        peers = self.node.easytier.get_peer_connections()
        
        print(f"\n🌐 网络连接 (共 {len(peers)} 个节点):")
        for peer in peers:
            status = "🟢" if peer.get('is_local') else "🟡"
            print(f"   {status} {peer['node_id']} - 延迟: {peer['latency']}ms")
    
    def security_management(self):
        """安全管理菜单"""
        while True:
            print("\n🛡️  安全管理")
            print("1. 查看黑白名单")
            print("2. 手动添加黑名单")
            print("3. 手动添加白名单") 
            print("4. 返回上级")
            
            choice = input("请选择操作 (1-4): ").strip()
            
            if choice == "1":
                print(f"   ✅ 白名单: {len(self.node.security.whitelist)} 个IP")
                print(f"   ❌ 黑名单: {len(self.node.security.blacklist)} 个IP")
                print(f"   ⚠️  可疑IP: {len(self.node.security.suspicious_ips)} 个")
            elif choice == "2":
                ip = input("请输入要添加的IP: ").strip()
                self.node.security.blacklist.add(ip)
                print(f"✅ 已添加黑名单: {ip}")
            elif choice == "3":
                ip = input("请输入要添加的IP: ").strip()
                self.node.security.whitelist.add(ip)
                print(f"✅ 已添加白名单: {ip}")
            elif choice == "4":
                break
            else:
                print("❌ 无效选择")
    
    def manual_sync(self):
        """手动同步集群"""
        print("🔄 正在同步集群状态...")
        self.node.load_security_lists()
        
        # 立即报告状态
        metrics = self.node.collect_metrics()
        encrypted_metrics = self.node.fernet.encrypt(
            json.dumps(metrics).encode()
        )
        
        try:
            response = requests.post(
                f"{self.node.config.worker_url}/api/nodes/metrics",
                headers={"X-Node-ID": self.node.config.node_id},
                data=encrypted_metrics,
                timeout=15
            )
            if response.status_code == 200:
                print("✅ 集群同步成功")
            else:
                print("❌ 集群同步失败")
        except Exception as e:
            print(f"❌ 同步错误: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="集群节点")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    config = NodeConfig(**config_data)
    node = ClusterNode(config)
    
    # 启动节点
    node.start()
    
    # 交互模式
    if args.interactive:
        cli = NodeCLI(node)
        cli.run()

if __name__ == "__main__":
    main()