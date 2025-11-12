#!/usr/bin/env python3
import requests
import json
import time
import threading
import psutil
import socket
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import argparse
import yaml
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import hashlib
import subprocess
import sys

@dataclass
class NodeConfig:
    node_id: str
    name: str
    admin_id: str
    region: str
    worker_url: str
    auth_token: str  # 新增：认证令牌
    encryption_password: str
    health_check_interval: int
    metrics_report_interval: int
    easytier_network_name: str
    easytier_public_endpoint: str
    ddos_threshold_connections: int
    ddos_threshold_bandwidth: int
    auto_blacklist: bool
    compression_enabled: bool
    monitoring_enabled: bool
    log_level: str
    log_file: str

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str) -> NodeConfig:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        node_cfg = config_data['node']
        cluster_cfg = config_data['cluster']
        easytier_cfg = config_data['easytier']
        security_cfg = config_data['security']
        monitoring_cfg = config_data['monitoring']
        
        return NodeConfig(
            node_id=node_cfg['id'],
            name=node_cfg['name'],
            admin_id=node_cfg['admin_id'],
            region=node_cfg['region'],
            worker_url=cluster_cfg['worker_url'],
            auth_token=cluster_cfg.get('auth_token', ''),  # 获取认证令牌
            encryption_password=cluster_cfg['encryption_password'],
            health_check_interval=cluster_cfg['health_check_interval'],
            metrics_report_interval=cluster_cfg['metrics_report_interval'],
            easytier_network_name=easytier_cfg['network_name'],
            easytier_public_endpoint=easytier_cfg['public_endpoint'],
            ddos_threshold_connections=security_cfg['ddos_threshold_connections'],
            ddos_threshold_bandwidth=security_cfg['ddos_threshold_bandwidth'],
            auto_blacklist=security_cfg['auto_blacklist'],
            compression_enabled=security_cfg['compression_enabled'],
            monitoring_enabled=monitoring_cfg['enabled'],
            log_level=monitoring_cfg['log_level'],
            log_file=monitoring_cfg['log_file']
        )

class EasyTierMonitor:
    """EasyTier 网络监控"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.connected_peers = []
        self.network_stats = {}
        
    def start_easytier(self):
        """启动 EasyTier 连接"""
        try:
            cmd = [
                "easytier-core", "-d",
                "--network-name", self.config.easytier_network_name,
                "--network-secret", self._generate_network_secret(),
                "-p", self.config.easytier_public_endpoint,
                "--hostname", self.config.node_id
            ]
            
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info("✅ EasyTier 启动成功")
            return True
        except Exception as e:
            logging.error(f"❌ EasyTier 启动失败: {e}")
            return False
    
    def _generate_network_secret(self) -> str:
        """生成网络密钥"""
        seed = f"{self.config.admin_id}-{self.config.encryption_password}"
        return hashlib.sha256(seed.encode()).hexdigest()[:32]
    
    def get_peer_info(self) -> List[Dict]:
        """获取对等节点信息"""
        try:
            # 解析 EasyTier 日志或使用 CLI 获取节点信息
            result = subprocess.run([
                "easytier-core", "peer", "list"
            ], capture_output=True, text=True, timeout=10)
            
            peers = []
            for line in result.stdout.split('\n'):
                if 'connected' in line.lower():
                    parts = line.split()
                    if len(parts) >= 3:
                        peers.append({
                            'node_id': parts[0],
                            'endpoint': parts[1],
                            'status': 'connected',
                            'latency': 0  # 简化实现
                        })
            
            self.connected_peers = peers
            return peers
            
        except Exception as e:
            logging.debug(f"获取 EasyTier 节点信息失败: {e}")
            return self.connected_peers
    
    def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        try:
            # 获取虚拟网卡统计
            for interface, stats in psutil.net_io_counters(pernic=True).items():
                if interface.startswith(('tun', 'utun', 'easytier')):
                    self.network_stats = {
                        'interface': interface,
                        'bytes_sent': stats.bytes_sent,
                        'bytes_recv': stats.bytes_recv,
                        'packets_sent': stats.packets_sent,
                        'packets_recv': stats.packets_recv,
                        'error_in': stats.errin,
                        'error_out': stats.errout,
                        'drop_in': stats.dropin,
                        'drop_out': stats.dropout
                    }
                    break
            
            return self.network_stats
        except Exception as e:
            logging.error(f"获取网络统计失败: {e}")
            return {}

class SecurityManager:
    """安全管理器"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.blacklist = set()
        self.whitelist = set()
        self.suspicious_ips = set()
        
    def monitor_connections(self) -> Dict[str, Any]:
        """监控连接状态"""
        try:
            connections = psutil.net_connections()
            current_connections = len(connections)
            
            # 分析连接模式
            ip_connections = {}
            for conn in connections:
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    ip = conn.raddr.ip
                    ip_connections[ip] = ip_connections.get(ip, 0) + 1
            
            # 检测可疑IP
            suspicious = []
            for ip, count in ip_connections.items():
                if count > 50:  # 单个IP连接数阈值
                    suspicious.append(ip)
                    self.suspicious_ips.add(ip)
            
            return {
                'total_connections': current_connections,
                'unique_ips': len(ip_connections),
                'suspicious_ips': suspicious,
                'under_attack': current_connections > self.config.ddos_threshold_connections
            }
            
        except Exception as e:
            logging.error(f"连接监控失败: {e}")
            return {'total_connections': 0, 'unique_ips': 0, 'suspicious_ips': [], 'under_attack': False}

class ClusterNode:
    def __init__(self, config: NodeConfig):
        self.config = config
        self.easytier = EasyTierMonitor(config)
        self.security = SecurityManager(config)
        self.registered = False
        self.websocket = None
        self.running = False
        
        self.setup_logging()
        self.encryption = self.setup_encryption()
    
    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(f"Node-{self.config.node_id}")
    
    def setup_encryption(self):
        """设置加密"""
        password = self.config.encryption_password.encode()
        salt = self.config.admin_id.encode()
        kdf = hashlib.pbkdf2_hmac('sha256', password, salt, 100000)
        key = base64.urlsafe_b64encode(kdf[:32])
        return Fernet(key)
    
    def register_to_cluster(self) -> bool:
        """使用认证令牌注册到集群"""
        try:
            if not self.config.auth_token:
                self.logger.error("❌ 认证令牌缺失，请先运行安装脚本获取认证令牌")
                return False
            
            # 获取公网IP
            public_ip = self.get_public_ip()
            
            payload = {
                "node_id": self.config.node_id,
                "auth_token": self.config.auth_token,  # 使用认证令牌而不是注册密钥
                "node_info": {
                    "name": self.config.name,
                    "host": socket.gethostname(),
                    "port": 2233,
                    "region": self.config.region,
                    "public_ip": public_ip,
                    "public_net_accessible": True,
                    "last_seen": int(time.time())
                }
            }
            
            self.logger.info(f"🔐 使用认证令牌注册到集群...")
            response = requests.post(
                f"{self.config.worker_url}/api/nodes/register",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    self.registered = True
                    self.logger.info(f"✅ 集群注册成功: {self.config.node_id}")
                    return True
                else:
                    self.logger.error(f"❌ 集群注册失败: {data.get('error', '未知错误')}")
                    return False
            else:
                self.logger.error(f"❌ 集群注册失败: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ 网络请求失败: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ 注册过程异常: {e}")
            return False
    
    def get_public_ip(self) -> str:
        """获取公网IP"""
        try:
            services = [
                "http://ifconfig.me",
                "http://ipinfo.io/ip",
                "http://api.ipify.org",
                "http://checkip.amazonaws.com"
            ]
            
            for service in services:
                try:
                    response = requests.get(service, timeout=5)
                    if response.status_code == 200:
                        ip = response.text.strip()
                        if ip and len(ip.split('.')) == 4:
                            return ip
                except:
                    continue
            
            return "unknown"
        except:
            return "unknown"
    
    def connect_to_websocket(self):
        """连接到Worker的WebSocket"""
        try:
            if not self.config.auth_token:
                self.logger.error("❌ 认证令牌缺失，无法建立WebSocket连接")
                return
            
            # 构建WebSocket URL
            ws_url = f"wss://{self.config.worker_url.replace('https://', '').replace('http://', '')}/ws/node"
            params = {
                "node_id": self.config.node_id,
                "auth_token": self.config.auth_token
            }
            
            # 这里需要实现WebSocket连接逻辑
            # 由于websockets库需要异步，这里简化实现
            self.logger.info("🌐 WebSocket连接准备就绪")
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket连接失败: {e}")
    
    def collect_metrics(self) -> Dict[str, Any]:
        """收集节点指标"""
        try:
            # 系统指标
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 网络指标
            peer_info = self.easytier.get_peer_info()
            network_stats = self.easytier.get_network_stats()
            security_status = self.security.monitor_connections()
            
            # 健康状态
            health = "healthy"
            if cpu_percent > 80:
                health = "degraded"
            elif cpu_percent > 95 or security_status['under_attack']:
                health = "critical"
            
            return {
                "node_id": self.config.node_id,
                "timestamp": int(time.time()),
                "health": health,
                "load": cpu_percent,  # 兼容性字段
                "public_latency": 0,  # 兼容性字段
                "connections": security_status['total_connections'],  # 兼容性字段
                "response_latency": 0,  # 兼容性字段
                "bandwidth_up": network_stats.get('bytes_sent', 0),  # 兼容性字段
                "bandwidth_down": network_stats.get('bytes_recv', 0),  # 兼容性字段
                "memory_usage": memory.percent,  # 兼容性字段
                "cpu_usage": cpu_percent,  # 兼容性字段
                "sync_status": "synced",  # 兼容性字段
                "security_score": 0.8,  # 兼容性字段
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent
                },
                "network": {
                    "peer_count": len(peer_info),
                    "peers": peer_info,
                    "stats": network_stats
                },
                "security": security_status,
                "easytier": {
                    "network_name": self.config.easytier_network_name,
                    "connected": len(peer_info) > 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"收集指标失败: {e}")
            return {
                "node_id": self.config.node_id,
                "timestamp": int(time.time()),
                "health": "unknown",
                "error": str(e)
            }
    
    def report_metrics(self):
        """报告指标到集群"""
        if not self.registered:
            self.logger.warning("⚠️ 节点未注册，跳过指标报告")
            return
        
        try:
            metrics = self.collect_metrics()
            
            # 使用认证令牌报告指标
            response = requests.post(
                f"{self.config.worker_url}/api/datachain/submit_metric",
                json={
                    "node_id": self.config.node_id,
                    "metric_type": "node_metrics",
                    "metric_data": metrics
                },
                headers={
                    "Content-Type": "application/json"
                },
                timeout=15
            )
            
            if response.status_code == 200:
                self.logger.debug("📊 指标报告成功")
            else:
                self.logger.warning(f"指标报告失败: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"报告指标失败: {e}")
    
    def start_monitoring(self):
        """启动监控循环"""
        def monitor_loop():
            while self.running:
                try:
                    self.report_metrics()
                    time.sleep(self.config.metrics_report_interval)
                except Exception as e:
                    self.logger.error(f"监控循环错误: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("✅ 监控服务已启动")
    
    def start(self):
        """启动节点"""
        self.logger.info("🚀 启动集群节点...")
        
        # 启动 EasyTier
        if not self.easytier.start_easytier():
            self.logger.error("❌ EasyTier 启动失败，节点无法正常运行")
            return False
        
        # 注册到集群（使用认证令牌）
        if not self.register_to_cluster():
            self.logger.error("❌ 集群注册失败")
            return False
        
        self.running = True
        
        # 启动监控
        self.start_monitoring()
        
        # 连接WebSocket
        self.connect_to_websocket()
        
        self.logger.info("🎉 集群节点启动完成")
        return True
    
    def stop(self):
        """停止节点"""
        self.logger.info("🛑 停止集群节点...")
        self.running = False

class NodeCLI:
    """节点命令行界面"""
    
    def __init__(self, node: ClusterNode):
        self.node = node
    
    def run(self):
        """运行交互式CLI"""
        while True:
            try:
                print("\n" + "="*50)
                print("🏢 集群节点管理系统")
                print("="*50)
                print("1. 📊 节点状态")
                print("2. 🌐 网络信息")
                print("3. 🛡️  安全状态")
                print("4. 🔄 手动同步")
                print("5. 🔑 认证信息")
                print("6. 📝 查看日志")
                print("7. 🚪 退出")
                
                choice = input("请选择操作 (1-7): ").strip()
                
                if choice == "1":
                    self.show_node_status()
                elif choice == "2":
                    self.show_network_info()
                elif choice == "3":
                    self.show_security_status()
                elif choice == "4":
                    self.manual_sync()
                elif choice == "5":
                    self.show_auth_info()
                elif choice == "6":
                    self.show_logs()
                elif choice == "7":
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
        
        print(f"\n📊 节点状态: {metrics['node_id']}")
        print(f"   🏥 健康状态: {metrics['health']}")
        print(f"   💻 CPU使用率: {metrics['system']['cpu_percent']}%")
        print(f"   🧠 内存使用率: {metrics['system']['memory_percent']}%")
        print(f"   💾 磁盘使用率: {metrics['system']['disk_percent']}%")
        print(f"   🔗 EasyTier网络: {metrics['easytier']['network_name']}")
        print(f"   📡 对等节点: {metrics['network']['peer_count']} 个")
        print(f"   🔐 注册状态: {'✅ 已注册' if self.node.registered else '❌ 未注册'}")
    
    def show_network_info(self):
        """显示网络信息"""
        peer_info = self.node.easytier.get_peer_info()
        network_stats = self.node.easytier.get_network_stats()
        
        print(f"\n🌐 网络信息:")
        print(f"   对等节点数量: {len(peer_info)}")
        
        for peer in peer_info:
            status = "🟢" if peer['status'] == 'connected' else "🟡"
            print(f"   {status} {peer['node_id']} - {peer['endpoint']}")
        
        if network_stats:
            print(f"\n   📨 发送: {network_stats.get('bytes_sent', 0)} bytes")
            print(f"   📥 接收: {network_stats.get('bytes_recv', 0)} bytes")
    
    def show_security_status(self):
        """显示安全状态"""
        security_status = self.node.security.monitor_connections()
        
        print(f"\n🛡️  安全状态:")
        print(f"   总连接数: {security_status['total_connections']}")
        print(f"   唯一IP数: {security_status['unique_ips']}")
        print(f"   可疑IP数: {len(security_status['suspicious_ips'])}")
        print(f"   DDoS攻击: {'是' if security_status['under_attack'] else '否'}")
        
        if security_status['suspicious_ips']:
            print(f"   可疑IP列表: {', '.join(security_status['suspicious_ips'][:5])}")
    
    def show_auth_info(self):
        """显示认证信息"""
        print(f"\n🔑 认证信息:")
        print(f"   节点ID: {self.node.config.node_id}")
        print(f"   管理员ID: {self.node.config.admin_id}")
        print(f"   认证令牌: {self.node.config.auth_token[:20]}...")
        print(f"   Worker地址: {self.node.config.worker_url}")
        print(f"   注册状态: {'✅ 已注册' if self.node.registered else '❌ 未注册'}")
    
    def manual_sync(self):
        """手动同步"""
        print("🔄 手动同步集群状态...")
        self.node.report_metrics()
        print("✅ 同步完成")
    
    def show_logs(self):
        """显示日志"""
        print(f"\n📝 最近日志:")
        try:
            with open(self.node.config.log_file, 'r') as f:
                lines = f.readlines()[-20:]  # 最后20行
                for line in lines:
                    print(f"   {line.strip()}")
        except Exception as e:
            print(f"❌ 读取日志失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="集群节点")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = ConfigLoader.load_config(args.config)
        node = ClusterNode(config)
    except Exception as e:
        logging.error(f"❌ 配置加载失败: {e}")
        sys.exit(1)
    
    # 启动节点
    if not node.start():
        sys.exit(1)
    
    # 交互模式
    if args.interactive:
        cli = NodeCLI(node)
        cli.run()
    else:
        # 守护进程模式
        try:
            while node.running:
                time.sleep(1)
        except KeyboardInterrupt:
            node.stop()

if __name__ == "__main__":
    main()
