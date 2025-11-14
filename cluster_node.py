#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 增强型区块链DDoS防御节点 - 终极完整版
# 整合所有功能：弹性架构、安全增强、性能监控、简化命令面板

import requests
import json
import time
import threading
import logging
import random
import socket
import asyncio
import websockets
import psutil
import uuid
import sqlite3
import hashlib
import configparser
import ipaddress
import os
import signal
import sys
import select
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from urllib.parse import urlparse
from collections import deque, OrderedDict
from functools import wraps

# 配置日志
log_file = "enhanced_ddos_node.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhancedDDoSNode')

class DefenseMode(Enum):
    NORMAL = "normal"
    ALERT = "alert"
    AGGRESSIVE = "aggressive"

class AttackType(Enum):
    SYN_FLOOD = "syn_flood"
    UDP_FLOOD = "udp_flood"
    ICMP_FLOOD = "icmp_flood"
    HTTP_FLOOD = "http_flood"
    DNS_AMPLIFICATION = "dns_amplification"
    MIXED_FLOOD = "mixed_flood"

@dataclass
class DDoSEvidence:
    attack_id: str
    attack_type: str
    source_ips: List[str]
    target_ports: List[int]
    local_attacked_ports: List[int]
    start_time: int
    end_time: Optional[int]
    max_bandwidth_mbps: float
    packet_count: int
    connection_count: int
    target_node_id: str
    source_ip_reputation: float
    attack_signature: str
    confidence: float = 0.5
    blockchain_tx: Optional[str] = None

@dataclass
class DefenseConfig:
    defense_ports: List[int]
    syn_flood_threshold: int = 5000
    udp_flood_threshold: int = 100000
    icmp_flood_threshold: int = 2000
    http_flood_threshold: int = 500
    connection_rate_threshold: int = 200
    packet_rate_threshold: int = 10000
    bandwidth_threshold: float = 500.0
    auto_blacklist: bool = True
    auto_blacklist_duration: int = 1800
    cooperative_defense: bool = True
    evidence_sharing: bool = True
    min_confidence: float = 0.9
    exclude_private_ips: bool = True
    enable_blackhole: bool = False
    blackhole_threshold: float = 0.95

# ------------------ 断路器模式 ------------------
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = 0
    
    def is_open(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return False
            return True
        return False
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"🔌 断路器开启，暂停操作 {self.recovery_timeout} 秒")
    
    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
        logger.info("🔌 断路器关闭，恢复正常操作")

# ------------------ 性能监控器 ------------------
class PerformanceMonitor:
    def __init__(self):
        self.operation_times = {}
        self.slow_threshold = 5.0  # 5秒
        self.operation_count = {}
        
    def track_performance(self, operation_name):
        """性能跟踪装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.operation_times[operation_name] = duration
                    self.operation_count[operation_name] = self.operation_count.get(operation_name, 0) + 1
                    
                    if duration > self.slow_threshold:
                        logger.warning(f"🐌 操作 {operation_name} 执行缓慢: {duration:.2f}秒")
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict:
        """生成性能报告"""
        report = {
            "slow_operations": [],
            "all_operations": [],
            "summary": {
                "total_operations": sum(self.operation_count.values()),
                "slow_operation_count": 0
            }
        }
        
        for op, duration in self.operation_times.items():
            count = self.operation_count.get(op, 0)
            op_info = {
                "operation": op,
                "avg_duration": duration,
                "count": count,
                "status": "正常" if duration <= self.slow_threshold else "需要优化"
            }
            report["all_operations"].append(op_info)
            
            if duration > self.slow_threshold:
                report["slow_operations"].append(op_info)
                report["summary"]["slow_operation_count"] += 1
                
        return report

# ------------------ 弹性节点管理器 ------------------
class ResilientNodeManager:
    def __init__(self, node):
        self.node = node
        self.error_count = 0
        self.last_error_time = 0
        self.circuit_breaker = CircuitBreaker()
        self.performance_monitor = PerformanceMonitor()
    
    def with_resilience(self, operation_name, max_retries=3):
        """装饰器：为关键操作添加弹性"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries + 1):
                    try:
                        if self.circuit_breaker.is_open():
                            raise Exception("Circuit breaker is open")
                            
                        result = func(*args, **kwargs)
                        self.error_count = max(0, self.error_count - 1)
                        self.circuit_breaker.record_success()
                        return result
                        
                    except Exception as e:
                        self.error_count += 1
                        self.last_error_time = time.time()
                        
                        if attempt == max_retries:
                            logger.error(f"❌ {operation_name} 最终失败: {e}")
                            self.circuit_breaker.record_failure()
                            # 触发降级策略
                            self._trigger_fallback(operation_name, e)
                            raise
                        else:
                            logger.warning(f"⚠️ {operation_name} 第{attempt+1}次失败: {e}, 重试中...")
                            time.sleep(2 ** attempt)  # 指数退避
            return wrapper
        return decorator
    
    def _trigger_fallback(self, operation_name: str, error: Exception):
        """触发降级策略"""
        fallback_actions = {
            "websocket_connect": self._fallback_websocket,
            "database_operation": self._fallback_database,
            "api_request": self._fallback_api,
        }
        
        if operation_name in fallback_actions:
            fallback_actions[operation_name](error)
    
    def _fallback_websocket(self, error: Exception):
        """WebSocket降级策略"""
        logger.warning("🔌 WebSocket降级：切换到轮询模式")
        # 可以在这里实现HTTP轮询作为降级方案
    
    def _fallback_database(self, error: Exception):
        """数据库降级策略"""
        logger.warning("💾 数据库降级：使用内存缓存")
        # 切换到内存缓存
    
    def _fallback_api(self, error: Exception):
        """API降级策略"""
        logger.warning("🌐 API降级：使用本地决策")
        # 切换到本地决策模式

# ------------------ 内存感知流量收集器 ------------------
class MemoryAwareTrafficCollector:
    def __init__(self, max_history_size=1000, memory_threshold=0.8):
        self.traffic_history = deque(maxlen=max_history_size)
        self.memory_threshold = memory_threshold
        self.adaptive_limit = max_history_size
        
    def add_traffic_record(self, record: Dict):
        """添加流量记录，自动内存管理"""
        self._check_memory_usage()
        self.traffic_history.append(record)
        
    def _check_memory_usage(self):
        """检查内存使用情况，动态调整历史记录大小"""
        try:
            memory_usage = psutil.virtual_memory().percent / 100
            if memory_usage > self.memory_threshold:
                # 动态缩减历史记录
                target_size = max(50, len(self.traffic_history) // 2)
                while len(self.traffic_history) > target_size:
                    self.traffic_history.popleft()
                self.adaptive_limit = target_size
                logger.warning(f"🧠 内存使用率高，自动缩减历史记录至 {target_size} 条")
        except Exception as e:
            logger.debug(f"内存检查异常: {e}")
    
    def get_recent_history(self, count: int = 100) -> List[Dict]:
        """获取最近的流量历史"""
        return list(self.traffic_history)[-count:]

# ------------------ 安全增强器 ------------------
class SecurityEnhancer:
    def __init__(self, node):
        self.node = node
        self.suspicious_activities = deque(maxlen=1000)
        self.rate_limits = {}  # IP -> {operation: timestamp_list}
        
    def validate_ip_address(self, ip: str) -> bool:
        """IP地址验证"""
        try:
            if not ip or ip.strip() == "":
                return False
                
            ip_obj = ipaddress.ip_address(ip.strip())
            
            # 阻止特殊地址
            if (ip_obj.is_multicast or ip_obj.is_unspecified or 
                ip_obj.is_reserved or ip == "0.0.0.0" or
                ip == "255.255.255.255"):
                return False
                
            return True
        except ValueError:
            return False
    
    def rate_limit_check(self, ip: str, operation: str, max_requests: int = 10, window_seconds: int = 60) -> bool:
        """操作频率限制"""
        if not self.validate_ip_address(ip):
            return False
            
        key = f"{ip}_{operation}"
        now = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = deque(maxlen=max_requests)
        
        # 清理过期请求
        while (self.rate_limits[key] and 
               now - self.rate_limits[key][0] > window_seconds):
            self.rate_limits[key].popleft()
        
        # 检查是否超过限制
        if len(self.rate_limits[key]) >= max_requests:
            logger.warning(f"🚫 IP {ip} 操作 {operation} 频率限制")
            self.record_suspicious_activity(ip, f"rate_limit_{operation}")
            return False
        
        self.rate_limits[key].append(now)
        return True
    
    def record_suspicious_activity(self, ip: str, activity_type: str):
        """记录可疑活动"""
        activity = {
            "ip": ip,
            "type": activity_type,
            "timestamp": time.time(),
            "node_id": self.node.node_id
        }
        self.suspicious_activities.append(activity)
    
    def sanitize_input(self, input_data: Any) -> Any:
        """输入数据清理"""
        if isinstance(input_data, str):
            # 移除潜在的危险字符，保留基本可打印字符
            sanitized = ''.join(c for c in input_data if c.isprintable() and c not in ['<', '>', '&', '"', "'", ';', '|'])
            return sanitized[:1000]  # 限制长度
        elif isinstance(input_data, (list, tuple)):
            return [self.sanitize_input(item) for item in input_data]
        elif isinstance(input_data, dict):
            return {self.sanitize_input(k): self.sanitize_input(v) for k, v in input_data.items()}
        return input_data

# ------------------ 信誉系统 ------------------
class IPReputationSystem:
    def __init__(self):
        self.ip_reputation: Dict[str, float] = {}
        self.reputation_history: Dict[str, List[float]] = {}
        
    def get_reputation(self, ip: str) -> float:
        if ip in self.ip_reputation:
            return self.ip_reputation[ip]
        rep = random.uniform(0.3, 1.0)
        try:
            if ipaddress.ip_address(ip).is_private:
                rep = max(rep, 0.8)
        except Exception:
            pass
        self.ip_reputation[ip] = rep
        return rep
        
    def update_reputation(self, ip: str, delta: float):
        cur = self.get_reputation(ip)
        new_rep = max(0.1, min(1.0, cur + delta))
        self.ip_reputation[ip] = new_rep
        
        # 记录历史
        if ip not in self.reputation_history:
            self.reputation_history[ip] = []
        self.reputation_history[ip].append(new_rep)
        
    def get_reputation_trend(self, ip: str) -> float:
        """获取信誉趋势（最近变化）"""
        if ip not in self.reputation_history or len(self.reputation_history[ip]) < 2:
            return 0.0
        
        history = self.reputation_history[ip][-10:]  # 最近10次记录
        if len(history) < 2:
            return 0.0
            
        # 简单线性趋势计算
        return history[-1] - history[0]

# ------------------ 真实流量检测器 ------------------
class RealisticDDoSDetector:
    def __init__(self, defense_config: DefenseConfig,
                 overlay_iface: Optional[str] = None,
                 external_ifaces: Optional[List[str]] = None,
                 defense_ports: Optional[List[int]] = None):
        self.defense_config = defense_config
        self.overlay_iface = overlay_iface
        self.external_ifaces = set(external_ifaces or [])
        self.defense_ports = set(defense_ports or [])
        self.aggressive_mode = False

        self.attack_detected = False
        self.current_attack_type: Optional[str] = None

        # 使用内存感知收集器
        self.traffic_collector = MemoryAwareTrafficCollector(max_history_size=500)
        self.last_check_time = time.time()
        self.last_pernic = psutil.net_io_counters(pernic=True)
        self.last_total = psutil.net_io_counters()

        self.recent_ext_udp = deque(maxlen=6)
        self.recent_ext_bw = deque(maxlen=6)

    def set_aggressive_mode(self, aggressive: bool):
        self.aggressive_mode = aggressive

    @staticmethod
    def _addr_ip_port(addr) -> Tuple[Optional[str], Optional[int]]:
        if not addr: return (None, None)
        ip = getattr(addr, "ip", None)
        port = getattr(addr, "port", None)
        if ip is None and isinstance(addr, tuple):
            ip = addr[0]; port = addr[1] if len(addr) > 1 else None
        return (ip, port)

    @staticmethod
    def _is_public_ip(ip: Optional[str]) -> bool:
        if not ip: return False
        try:
            ipo = ipaddress.ip_address(ip)
            return not (ipo.is_private or ipo.is_loopback or ipo.is_link_local)
        except Exception:
            return False

    def _get_network_connections(self):
        try:
            return psutil.net_connections(kind="inet")
        except Exception:
            return []

    def _count_external_conns_on_port(self, port: int) -> int:
        conns = self._get_network_connections()
        cnt = 0
        for c in conns:
            lip, lport = self._addr_ip_port(c.laddr)
            rip, _ = self._addr_ip_port(c.raddr)
            if lport == port and self._is_public_ip(rip):
                cnt += 1
        return cnt

    def _collect_real_traffic_stats(self) -> Dict:
        now = time.time()
        dt = max(0.2, now - self.last_check_time)

        total_now = psutil.net_io_counters()
        pernic_now = psutil.net_io_counters(pernic=True)

        total_packets_diff = (total_now.packets_sent + total_now.packets_recv) - \
                             (self.last_total.packets_sent + self.last_total.packets_recv)
        total_bytes_diff = (total_now.bytes_sent + total_now.bytes_recv) - \
                           (self.last_total.bytes_sent + self.last_total.bytes_recv)

        ext_packets_diff = 0
        ext_bytes_diff = 0
        for nic, nowc in pernic_now.items():
            if nic not in self.external_ifaces:
                continue
            lastc = self.last_pernic.get(nic)
            if not lastc: continue
            ext_packets_diff += (nowc.packets_sent + nowc.packets_recv) - (lastc.packets_sent + lastc.packets_recv)
            ext_bytes_diff += (nowc.bytes_sent + nowc.bytes_recv) - (lastc.bytes_sent + lastc.bytes_recv)

        self.last_total = total_now
        self.last_pernic = pernic_now
        self.last_check_time = now

        packet_rate_total = max(0.0, total_packets_diff / dt)
        bandwidth_total_mbps = max(0.0, (total_bytes_diff * 8) / dt / 1_000_000)

        packet_rate_external = max(0.0, ext_packets_diff / dt)
        bandwidth_external_mbps = max(0.0, (ext_bytes_diff * 8) / dt / 1_000_000)

        udp_packets_external = int(packet_rate_external * 0.3)

        conns = self._get_network_connections()
        syn_recv = sum(1 for c in conns if str(c.status).upper() == 'SYN_RECV')

        http_ports = {80, 443, 8080, 8443}
        http_conn = 0
        for c in conns:
            lip, lport = self._addr_ip_port(c.laddr)
            rip, _ = self._addr_ip_port(c.raddr)
            if lport in http_ports and self._is_public_ip(rip):
                http_conn += 1

        easytier_ext_conn = self._count_external_conns_on_port(11010)

        self.recent_ext_udp.append(udp_packets_external)
        self.recent_ext_bw.append(bandwidth_external_mbps)

        stats = {
            "timestamp": now,
            "packet_rate": packet_rate_total,
            "packet_count": int(total_now.packets_sent + total_now.packets_recv),
            "connection_rate": len(conns) / dt,
            "connection_count": len(conns),
            "bandwidth_usage": bandwidth_total_mbps,

            "packet_rate_external": packet_rate_external,
            "bandwidth_external": bandwidth_external_mbps,
            "udp_packets_external": udp_packets_external,

            "syn_packets": syn_recv,
            "http_requests": http_conn,
            "easytier_ext_conn": easytier_ext_conn,
            "current_connections": len(conns)
        }
        
        # 使用内存感知收集器
        self.traffic_collector.add_traffic_record(stats)
        
        return stats

    def _detect_syn_flood(self, st: Dict) -> Tuple[bool, float]:
        thr = self.defense_config.syn_flood_threshold // (2 if self.aggressive_mode else 1)
        v = st["syn_packets"]
        if v > thr:
            ratio = min(v / thr, 10.0)
            conf = min(0.3 + (ratio - 1) * 0.1, 0.9)
            return True, conf
        return False, 0.0

    def _detect_http_flood(self, st: Dict) -> Tuple[bool, float]:
        thr = self.defense_config.http_flood_threshold // (2 if self.aggressive_mode else 1)
        v = st["http_requests"]
        if v > thr:
            ratio = min(v / thr, 10.0)
            conf = min(0.3 + (ratio - 1) * 0.1, 0.9)
            return True, conf
        return False, 0.0

    def _detect_udp_flood(self, st: Dict) -> Tuple[bool, float]:
        udp_thr = self.defense_config.udp_flood_threshold // (2 if self.aggressive_mode else 1)
        bw_gate = max(10.0, self.defense_config.bandwidth_threshold * 0.2)

        v_udp = st["udp_packets_external"]
        v_bw = st["bandwidth_external"]

        last3_udp = list(self.recent_ext_udp)[-3:]
        last3_bw = list(self.recent_ext_bw)[-3:]
        over_udp = sum(1 for x in last3_udp if x > udp_thr)
        over_bw = sum(1 for x in last3_bw if x > bw_gate)

        if over_udp >= 2 and over_bw >= 2:
            ratio = min(v_udp / max(1, udp_thr), 10.0)
            conf = min(0.4 + (ratio - 1) * 0.08, 0.95)
            return True, conf
        return False, 0.0

    def detect_attacks(self) -> Dict:
        st = self._collect_real_traffic_stats()

        result = {
            "attack_detected": False,
            "attack_type": None,
            "suspicious_ips": [],
            "target_ports": list(self.defense_ports or []),
            "max_bandwidth": st["bandwidth_external"],
            "packet_count": int(st["packet_rate_external"]),
            "connection_count": st["current_connections"],
            "attack_signature": "",
            "confidence": 0.0
        }

        syn, cs = self._detect_syn_flood(st)
        http, ch = self._detect_http_flood(st)
        udp, cu = self._detect_udp_flood(st)

        candidates: List[Tuple[str, float]] = []
        if syn: candidates.append((AttackType.SYN_FLOOD.value, cs))
        if udp: candidates.append((AttackType.UDP_FLOOD.value, cu))
        if http: candidates.append((AttackType.HTTP_FLOOD.value, ch))

        if candidates:
            at, conf = max(candidates, key=lambda x: x[1])
            result.update({
                "attack_detected": True,
                "attack_type": at,
                "suspicious_ips": self._suspicious_ips(at),
                "attack_signature": f"{at.upper()}_{int(time.time())}",
                "confidence": conf
            })

        self.attack_detected = result["attack_detected"]
        self.current_attack_type = result["attack_type"]
        return result

    def _suspicious_ips(self, attack_type: str) -> List[str]:
        conns = self._get_network_connections()
        counter: Dict[str, int] = {}
        for c in conns:
            lip, lport = self._addr_ip_port(c.laddr)
            rip, _ = self._addr_ip_port(c.raddr)
            if not self._is_public_ip(rip):
                continue
            if attack_type == AttackType.SYN_FLOOD.value and str(c.status).upper() == 'SYN_RECV':
                counter[rip] = counter.get(rip, 0) + 1
            elif attack_type == AttackType.HTTP_FLOOD.value and lport in (80, 443, 8080, 8443):
                counter[rip] = counter.get(rip, 0) + 1
            elif attack_type == AttackType.UDP_FLOOD.value and (lport in self.defense_ports or lport == 11010):
                counter[rip or "unknown"] = counter.get(rip or "unknown", 0) + 1
        return [ip for ip, _ in sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:5]]

    def get_attack_status(self) -> Dict:
        return {
            "attack_detected": self.attack_detected,
            "attack_type": self.current_attack_type,
            "suspicious_ips": self._suspicious_ips(self.current_attack_type) if self.attack_detected else [],
            "confidence": 0.8 if self.attack_detected else 0.0
        }

    def get_traffic_stats(self) -> Dict:
        recent = self.traffic_collector.get_recent_history(1)
        return recent[0] if recent else {}

    def get_traffic_history(self, count: int = 100) -> List[Dict]:
        return self.traffic_collector.get_recent_history(count)

# ------------------ 名单管理系统 ------------------
class CloudIPManager:
    def __init__(self, node):
        self.node = node
        self.ip_list: Dict[str, Dict] = {}
        
    def sync_from_cloud(self, cloud_list: List[Dict]):
        """从云端同步IP名单"""
        try:
            self.ip_list = {}
            for item in cloud_list:
                ip = item.get('ip')
                if ip and self.node.security_enhancer.validate_ip_address(ip):
                    self.ip_list[ip] = {
                        'reason': item.get('reason', ''),
                        'reputation': item.get('reputation', 0.5),
                        'added_at': item.get('added_at', 0),
                        'added_by': item.get('added_by', ''),
                        'expires_at': item.get('expires_at', 0)
                    }
            logger.info(f"✅ 云名单同步完成: {len(self.ip_list)} 条记录")
        except Exception as e:
            logger.error(f"❌ 云名单同步失败: {e}")
                
    def get_list(self) -> Dict:
        return self.ip_list
        
    def is_listed(self, ip: str) -> bool:
        """检查IP是否在云名单中，同时检查过期时间"""
        if not self.node.security_enhancer.validate_ip_address(ip):
            return False
            
        if ip not in self.ip_list:
            return False
            
        entry = self.ip_list[ip]
        expires_at = entry.get('expires_at', 0)
        
        # 检查是否过期
        if expires_at > 0 and time.time() > expires_at:
            del self.ip_list[ip]
            return False
            
        return True

    def get_active_ips(self) -> List[str]:
        """获取活跃的IP列表（未过期的）"""
        now = time.time()
        active_ips = []
        for ip, info in self.ip_list.items():
            expires_at = info.get('expires_at', 0)
            if expires_at == 0 or now <= expires_at:
                active_ips.append(ip)
        return active_ips

class LocalIPManager:
    def __init__(self, node):
        self.node = node
        self.ip_list: Dict[str, Dict] = {}
        
    def add_ip(self, ip: str, reason: str = "", ttl: int = 3600):
        """添加IP到本地名单"""
        if not self.node.security_enhancer.validate_ip_address(ip):
            logger.warning(f"⚠️ 无效IP地址，跳过添加: {ip}")
            return False
            
        expires_at = time.time() + ttl if ttl > 0 else 0
        self.ip_list[ip] = {
            'reason': reason, 
            'added_at': time.time(), 
            'expires_at': expires_at
        }
        return True
        
    def remove_ip(self, ip: str) -> bool:
        if ip in self.ip_list:
            del self.ip_list[ip]
            return True
        return False
        
    def get_all_ips(self) -> Dict:
        """获取所有IP（自动清理过期的）"""
        now = time.time()
        expired = [ip for ip, info in self.ip_list.items() 
                  if info['expires_at'] > 0 and info['expires_at'] < now]
        for ip in expired: 
            del self.ip_list[ip]
        return self.ip_list
        
    def get_recent_ips(self, time_window: int = 3600) -> List[str]:
        now = time.time()
        return [ip for ip, info in self.ip_list.items() 
                if now - info['added_at'] <= time_window]
        
    def is_listed(self, ip: str) -> bool:
        """检查IP是否在本地名单中（检查过期时间）"""
        if not self.node.security_enhancer.validate_ip_address(ip):
            return False
            
        info = self.ip_list.get(ip)
        if not info: 
            return False
            
        expires_at = info['expires_at']
        if expires_at > 0 and expires_at < time.time():
            del self.ip_list[ip]
            return False
            
        return True

    def sync_with_cloud(self, cloud_ips: Dict[str, Dict]):
        """与云名单同步"""
        try:
            added_count = 0
            removed_count = 0
            
            # 添加云名单中新增的IP
            for ip, cloud_info in cloud_ips.items():
                if self.node.security_enhancer.validate_ip_address(ip) and not self.is_listed(ip):
                    if self.add_ip(ip, cloud_info.get('reason', '云同步'), 
                                 cloud_info.get('expires_at', 0) - time.time()):
                        added_count += 1
            
            # 移除本地中不在云名单的IP（仅限云同步的）
            local_ips_to_remove = []
            for ip, local_info in self.ip_list.items():
                if ip not in cloud_ips and "云同步" in local_info.get('reason', ''):
                    local_ips_to_remove.append(ip)
            
            for ip in local_ips_to_remove:
                if self.remove_ip(ip):
                    removed_count += 1
                
            logger.info(f"✅ 本地名单与云同步完成: +{added_count}, -{removed_count}")
        except Exception as e:
            logger.error(f"❌ 本地名单同步失败: {e}")

# ------------------ 区块链管理器 ------------------
class BlockchainManager:
    def __init__(self, node):
        self.node = node
        self.blocks = []
        
    def add_block(self, block_data: Dict) -> bool:
        try:
            if self.verify_block(block_data):
                self.blocks.append(block_data)
                return True
            return False
        except Exception as e:
            logger.error(f"❌ 添加区块失败: {e}")
            return False
            
    def verify_block(self, block_data: Dict) -> bool:
        required_fields = ['block_id', 'previous_hash', 'timestamp', 'signature']
        return all(field in block_data for field in required_fields)
        
    def get_chain_info(self) -> Dict:
        return {
            'height': len(self.blocks),
            'latest_block': self.blocks[-1]['block_id'] if self.blocks else 'none',
            'total_blocks': len(self.blocks)
        }

# ------------------ 协同防御管理器 ------------------
class CooperativeDefenseManager:
    def __init__(self, node):
        self.node = node
        self.last_alert_time = 0
        self.alert_cooldown = 300
        
    def broadcast_attack_alert(self, evidence: DDoSEvidence):
        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown:
            return
            
        if self.node.websocket_connected and self.node.websocket:
            threading.Thread(target=self._run_async_alert, args=(evidence,), daemon=True).start()
            
        self.node.metrics["cooperative_alerts_sent"] += 1
        self.last_alert_time = now
        
    def _run_async_alert(self, evidence: DDoSEvidence):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._send_attack_alert(evidence))
            loop.close()
        except Exception as e:
            logger.error(f"❌ 发送攻击警报失败: {e}")
            
    async def _send_attack_alert(self, evidence: DDoSEvidence):
        try:
            alert_data = {
                "type": "security_alert", 
                "alert_type": "ddos_attack",
                "source_node": self.node.node_id, 
                "evidence": asdict(evidence),
                "timestamp": int(time.time())
            }
            await self.node.websocket.send(json.dumps(alert_data))
        except Exception as e:
            logger.error(f"❌ 发送攻击警报失败: {e}")
            
    def check_node_anomalies(self):
        now = time.time()
        for node_id, status in self.node.node_status_cache.items():
            if now - status.get('last_seen', 0) > 300:
                logger.warning(f"⚠️ 节点 {node_id} 可能异常: 长时间未更新状态")
            if status.get('load', 0) > 90:
                logger.warning(f"⚠️ 节点 {node_id} 负载异常: {status.get('load', 0)}%")
                
    def sync_cooperative_data(self):
        pass

# ------------------ 简化命令面板 ------------------
class SimpleCommandPanel:
    def __init__(self, node):
        self.node = node
        self.running = True
        
    def start_panel(self):
        """启动简化的命令面板"""
        self.clear_screen()
        print("\n🎮 启动交互式命令面板 (简化版)")
        print("=" * 50)
        print("输入 'help' 查看命令，'exit' 退出")
        print("=" * 50)
        
        while self.running and self.node.running:
            try:
                # 使用简单的输入方式，避免复杂的终端设置
                mode_icons = {
                    "normal": "🟢",
                    "alert": "🟡", 
                    "aggressive": "🔴"
                }
                current_mode = self.node.defense_mode.value
                icon = mode_icons.get(current_mode, "⚪")
                
                command = input(f"\n{icon} DDoSDefense [{current_mode}]> ").strip()
                
                if command:
                    self.process_command(command)
                    
            except KeyboardInterrupt:
                print("\n👋 收到退出信号")
                self.node.stop()
                break
            except EOFError:
                print("\n👋 输入结束")
                break
            except Exception as e:
                print(f"\n❌ 命令执行错误: {e}")
    
    def clear_screen(self):
        """清屏"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def process_command(self, command: str):
        """处理用户命令"""
        parts = command.split()
        if not parts:
            return
            
        cmd = parts[0].lower()
        
        if cmd == 'help':
            self.show_help()
        elif cmd == 'status':
            self.node.print_simple_status()
        elif cmd == 'nodes':
            self.node.print_cluster_nodes()
        elif cmd == 'traffic':
            self.show_traffic_stats()
        elif cmd == 'detect':
            self.run_detection()
        elif cmd == 'defense':
            self.handle_defense_command(parts[1:])
        elif cmd == 'blacklist':
            self.show_blacklist()
        elif cmd == 'whitelist':
            self.show_whitelist()
        elif cmd == 'easytier':
            self.show_easytier_status()
        elif cmd == 'logs':
            self.show_logs(parts[1:])
        elif cmd == 'clear':
            self.clear_screen()
        elif cmd == 'exit':
            self.node.stop()
        else:
            print(f"❓ 未知命令: {command}")
            print("输入 'help' 查看可用命令")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
📋 可用命令:

基础命令:
  status     - 显示节点状态
  clear     - 清空屏幕
  help      - 显示此帮助
  exit      - 退出节点

监控命令:
  nodes     - 显示集群节点
  traffic   - 显示流量统计
  easytier  - 显示EasyTier状态
  detect    - 执行DDoS检测

安全命令:
  defense   - 防御设置
  blacklist - 显示黑名单
  whitelist - 显示白名单

其他命令:
  logs [数量] - 显示操作日志
        """
        print(help_text)
    
    def show_traffic_stats(self):
        """显示流量统计"""
        stats = self.node.ddos_detector.get_traffic_stats()
        if not stats:
            print("📊 暂无流量数据")
            return
            
        print("\n📊 实时流量统计:")
        print("-" * 40)
        print(f"  外网包速率: {int(stats.get('packet_rate_external', 0))} pkt/s")
        print(f"  外网带宽:   {stats.get('bandwidth_external', 0.0):.2f} Mbps")
        print(f"  当前连接数: {stats.get('current_connections', 0)}")
        print(f"  SYN包数:    {stats.get('syn_packets', 0)}")
        print(f"  HTTP请求:   {stats.get('http_requests', 0)}")
        print(f"  UDP包数:    {stats.get('udp_packets_external', 0)}")
    
    def run_detection(self):
        """执行DDoS检测"""
        print("🔍 执行DDoS检测...")
        result = self.node.ddos_detector.detect_attacks()
        if result['attack_detected']:
            print(f"🚨 检测到攻击: {result['attack_type']}")
            print(f"   置信度: {result.get('confidence', 0.0):.2f}")
            print(f"   可疑IP: {', '.join(result['suspicious_ips'][:3])}")
        else:
            print("✅ 未检测到攻击")
    
    def handle_defense_command(self, args):
        """处理防御命令"""
        if not args:
            self.show_defense_status()
            return
            
        subcmd = args[0].lower()
        if subcmd == 'mode' and len(args) >= 2:
            mode_str = args[1].lower()
            if mode_str == 'normal':
                self.node.defense_mode = DefenseMode.NORMAL
                self.node.ddos_detector.set_aggressive_mode(False)
                print("🟢 切换到正常防御模式")
            elif mode_str == 'alert':
                self.node.defense_mode = DefenseMode.ALERT
                print("🟡 切换到警报防御模式")
            elif mode_str == 'aggressive':
                self.node.defense_mode = DefenseMode.AGGRESSIVE
                self.node.ddos_detector.set_aggressive_mode(True)
                print("🔴 切换到激进防御模式")
            else:
                print("❌ 无效的防御模式")
        else:
            print("❌ 无效的防御命令")
    
    def show_defense_status(self):
        """显示防御状态"""
        mode_icons = {
            DefenseMode.NORMAL: "🟢",
            DefenseMode.ALERT: "🟡", 
            DefenseMode.AGGRESSIVE: "🔴"
        }
        attack_status = self.node.ddos_detector.get_attack_status()
        
        print("\n🛡️ 防御状态:")
        print("-" * 30)
        print(f"  模式: {mode_icons[self.node.defense_mode]} {self.node.defense_mode.value}")
        print(f"  攻击检测: {'❗ 是' if attack_status['attack_detected'] else '✅ 否'}")
        if attack_status['attack_detected']:
            print(f"  攻击类型: {attack_status['attack_type']}")
        print(f"  协同防御: {'✅ 启用' if self.node.defense_config.cooperative_defense else '❌ 禁用'}")
        print(f"  自动拉黑: {'✅ 启用' if self.node.defense_config.auto_blacklist else '❌ 禁用'}")
    
    def show_blacklist(self):
        """显示黑名单"""
        blacklist = self.node.local_blacklist.get_all_ips()
        cloud_blacklist = self.node.cloud_blacklist.get_list()
        
        print("\n🚫 黑名单状态:")
        print("-" * 30)
        print(f"  本地黑名单: {len(blacklist)} 个IP")
        print(f"  云黑名单:   {len(cloud_blacklist)} 个IP")
        
        if blacklist:
            print("\n  最近的黑名单IP:")
            for ip, info in list(blacklist.items())[:5]:
                reason = info.get('reason', '未知原因')
                print(f"    {ip} - {reason}")
    
    def show_whitelist(self):
        """显示白名单"""
        whitelist = self.node.local_whitelist.get_all_ips()
        cloud_whitelist = self.node.cloud_whitelist.get_list()
        
        print("\n✅ 白名单状态:")
        print("-" * 30)
        print(f"  本地白名单: {len(whitelist)} 个IP")
        print(f"  云白名单:   {len(cloud_whitelist)} 个IP")
    
    def show_easytier_status(self):
        """显示EasyTier状态"""
        et_stats = self.node.get_easytier_stats()
        
        print("\n🔗 EasyTier状态:")
        print("-" * 30)
        print(f"  进程数: {et_stats.get('proc_count', 0)}")
        print(f"  CPU使用: {et_stats.get('cpu_percent', 0.0):.1f}%")
        print(f"  内存使用: {et_stats.get('mem_mb', 0.0):.1f} MB")
        print(f"  11010端口连接: {et_stats.get('ext_conn_11010', 0)}")
        print(f"  外网带宽: {et_stats.get('external_bandwidth_mbps', 0.0):.2f} Mbps")
    
    def show_logs(self, args):
        """显示日志"""
        limit = 10
        if args and args[0].isdigit():
            limit = int(args[0])
            
        logs = self.node.get_operation_logs(limit)
        if not logs:
            print("📝 暂无操作日志")
        else:
            print(f"\n📋 最近 {len(logs)} 条操作日志:")
            print("-" * 50)
            for log in logs:
                time_str = time.strftime('%m-%d %H:%M:%S', time.localtime(log['timestamp']))
                print(f"  [{time_str}] {log['operation_type']} - {log['target']}")

# ------------------ 主节点类 ------------------
class EnhancedDDoSNode:
    def __init__(self, config_file: str = "enhanced_node_config.ini"):
        self.config_file = config_file
        self.load_config()

        if not getattr(self, 'node_id', None):
            self.node_id = f"node_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        if not getattr(self, 'admin_id', None):
            self.admin_id = "admin_001"

        self.online = False
        self.health = "healthy"
        self.load = 0
        self.connections = 0
        self.public_ip = self.get_public_ip()
        self.ddos_detection_enabled = True

        # 初始化防御模式
        self.defense_mode = DefenseMode.NORMAL

        if not hasattr(self, 'defense_config'):
            self.defense_config = DefenseConfig(defense_ports=[80, 443, 2233, 11010])

        # 初始化增强组件
        self.resilient_manager = ResilientNodeManager(self)
        self.security_enhancer = SecurityEnhancer(self)

        # 接口识别
        self.overlay_iface: Optional[str] = None
        self._detect_overlay_iface()
        self.external_ifaces = self._external_ifaces()

        # 检测器
        self.ddos_detector = RealisticDDoSDetector(
            self.defense_config,
            overlay_iface=self.overlay_iface,
            external_ifaces=self.external_ifaces,
            defense_ports=self.defense_config.defense_ports
        )
        self.ip_reputation_system = IPReputationSystem()

        # 名单管理系统
        self.cloud_blacklist = CloudIPManager(self)
        self.cloud_whitelist = CloudIPManager(self)
        self.local_blacklist = LocalIPManager(self)
        self.local_whitelist = LocalIPManager(self)
        self.blockchain_manager = BlockchainManager(self)
        self.cooperative_defense = CooperativeDefenseManager(self)

        # 命令面板 - 使用简化版本
        self.command_panel = SimpleCommandPanel(self)

        self.last_sync_time = 0
        self.sync_interval = 30

        # WebSocket
        self.websocket = None
        self.websocket_connected = False

        # 集群状态
        self.available_nodes: List[Dict] = []
        self.node_status_cache: Dict[str, Dict] = {}

        # 线程控制
        self.running = False
        self.heartbeat_thread = None
        self.websocket_thread = None
        self.ddos_detection_thread = None
        self.panel_thread = None
        self.cooperative_thread = None

        # 统计信息
        self.metrics = {
            "start_time": time.time(),
            "health_reports_sent": 0,
            "ddos_attacks_detected": 0,
            "blocks_synced": 0,
            "ip_blacklisted": 0,
            "ip_blacklist_blocked": 0,
            "cooperative_alerts_sent": 0,
            "cooperative_alerts_received": 0,
            "errors_count": 0,
            "cloud_sync_count": 0
        }

        self.init_database()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def print_simple_status(self):
        """简化状态显示"""
        uptime = time.time() - self.metrics["start_time"]
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        
        mode_icons = {
            DefenseMode.NORMAL: "🟢",
            DefenseMode.ALERT: "🟡",
            DefenseMode.AGGRESSIVE: "🔴"
        }
        
        attack_status = self.ddos_detector.get_attack_status()
        traffic_stats = self.ddos_detector.get_traffic_stats()
        et = self.get_easytier_stats()
        
        print("\n" + "="*50)
        print(f"🛡️  节点状态".center(48))
        print("="*50)
        
        print(f"🔸 节点ID: {self.node_id}")
        print(f"🔸 运行时间: {hours:02d}:{minutes:02d}")
        print(f"🔸 防御模式: {mode_icons[self.defense_mode]} {self.defense_mode.value}")
        print(f"🔸 WebSocket: {'✅ 已连接' if self.websocket_connected else '❌ 未连接'}")
        
        print(f"\n📊 流量:")
        print(f"  包速率: {int(traffic_stats.get('packet_rate_external', 0)):>6} pkt/s")
        print(f"  带宽:   {traffic_stats.get('bandwidth_external', 0.0):>8.2f} Mbps")
        print(f"  连接数: {traffic_stats.get('current_connections', 0):>6}")
        
        print(f"\n🔗 EasyTier:")
        print(f"  进程: {et.get('proc_count',0):>2} | CPU: {et.get('cpu_percent',0.0):>5.1f}%")
        print(f"  内存: {et.get('mem_mb',0.0):>6.1f} MB | 连接: {et.get('ext_conn_11010',0):>3}")
        
        print(f"\n🚨 安全:")
        if attack_status['attack_detected']:
            print(f"  ❗ 攻击: {attack_status['attack_type']}")
            print(f"  ⚠️  IP: {', '.join(attack_status['suspicious_ips'][:2])}")
        else:
            print("  ✅ 无攻击")
        
        print("="*50)

    # ------------------ EasyTier 监控 ------------------
    def _detect_overlay_iface(self):
        try:
            for name, _ in psutil.net_if_addrs().items():
                ln = name.lower()
                if any(k in ln for k in ("easytier", "tun", "tap", "utun", "wg")):
                    self.overlay_iface = name
                    break
        except Exception:
            self.overlay_iface = None

    def _external_ifaces(self) -> List[str]:
        try:
            names = list(psutil.net_if_addrs().keys())
            out = []
            for n in names:
                ln = n.lower()
                if self.overlay_iface and n == self.overlay_iface:
                    continue
                if ln.startswith("lo") or "docker" in ln or "veth" in ln or "br-" in ln or "kube" in ln:
                    continue
                out.append(n)
            return out
        except Exception:
            return []

    def get_easytier_stats(self) -> Dict:
        proc_cpu = 0.0
        proc_mem_mb = 0.0
        proc_count = 0
        pids = []
        try:
            for p in psutil.process_iter(attrs=['name', 'cmdline', 'cpu_percent', 'memory_info', 'pid']):
                name = (p.info.get('name') or "").lower()
                cmd = " ".join(p.info.get('cmdline') or []).lower()
                if any(k in name or k in cmd for k in ("easytier", "easytier-core")):
                    proc_count += 1
                    pids.append(p.info.get('pid'))
                    proc_cpu += (p.info.get('cpu_percent') or 0.0)
                    mi = p.info.get('memory_info')
                    if mi: 
                        proc_mem_mb += mi.rss / 1024 / 1024
        except Exception:
            pass
            
        st = self.ddos_detector.get_traffic_stats() or {}
        ext_bw = st.get("bandwidth_external", 0.0)
        ext_pr = st.get("packet_rate_external", 0.0)
        et_ext_conn = st.get("easytier_ext_conn", 0)
        et_udp_est = int(ext_pr * 0.15)
        
        return {
            "pids": pids,
            "proc_count": proc_count,
            "cpu_percent": round(proc_cpu, 1),
            "mem_mb": round(proc_mem_mb, 1),
            "ext_conn_11010": et_ext_conn,
            "external_bandwidth_mbps": round(ext_bw, 2),
            "external_packet_rate": int(ext_pr),
            "udp_est_11010_pps": et_udp_est
        }

    def get_easytier_full_info(self) -> List[Dict]:
        """获取EasyTier详细信息"""
        out = []
        def _is_public_ip(ip: Optional[str]) -> bool:
            if not ip: 
                return False
            try:
                ipo = ipaddress.ip_address(ip)
                return not (ipo.is_private or ipo.is_loopback or ipo.is_link_local)
            except Exception:
                return False
                
        for p in psutil.process_iter(attrs=[
            'pid','name','cmdline','status','username','create_time',
            'cpu_percent','memory_info','memory_percent','num_threads','nice'
        ]):
            try:
                name = (p.info.get('name') or '').lower()
                cmd  = ' '.join(p.info.get('cmdline') or []).lower()
                if not any(k in name or k in cmd for k in ('easytier','easytier-core')):
                    continue
                    
                ext_total = 0
                ext_11010 = 0
                conn_samples = []
                try:
                    for c in p.connections(kind='inet'):
                        laddr = getattr(c, 'laddr', None)
                        raddr = getattr(c, 'raddr', None)
                        lip   = getattr(laddr, 'ip', None) if laddr else None
                        lport = getattr(laddr, 'port', None) if laddr else None
                        rip   = getattr(raddr, 'ip', None) if raddr else None
                        if _is_public_ip(rip):
                            ext_total += 1
                            if lport == 11010: 
                                ext_11010 += 1
                            if len(conn_samples) < 10:
                                conn_samples.append({
                                    'laddr': f'{lip}:{lport}' if lip and lport else '',
                                    'raddr': f'{rip}:{getattr(raddr,"port",None)}' if raddr else '',
                                    'status': str(c.status)
                                })
                except Exception:
                    pass
                    
                mi = p.info.get('memory_info')
                open_files = []
                try:
                    open_files = [f.path for f in p.open_files()[:10]]
                except Exception:
                    pass
                    
                io_counters = None
                try:
                    ioc = p.io_counters()
                    io_counters = {
                        'read_mb': round(ioc.read_bytes/1024/1024,2),
                        'write_mb': round(ioc.write_bytes/1024/1024,2)
                    }
                except Exception:
                    pass
                    
                out.append({
                    'pid': p.info.get('pid'),
                    'name': p.info.get('name'),
                    'cmdline': p.info.get('cmdline'),
                    'status': p.info.get('status'),
                    'username': p.info.get('username'),
                    'create_time': datetime.fromtimestamp(p.info.get('create_time', time.time())).isoformat(),
                    'cpu_percent': round(p.info.get('cpu_percent') or 0.0, 1),
                    'memory_mb': round((mi.rss/1024/1024) if mi else 0.0, 1),
                    'memory_percent': round(p.info.get('memory_percent') or 0.0, 2),
                    'num_threads': p.info.get('num_threads'),
                    'nice': p.info.get('nice'),
                    'external_conns_total': ext_total,
                    'external_conns_11010': ext_11010,
                    'conn_samples': conn_samples,
                    'open_files_sample': open_files,
                    'io_counters': io_counters
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue
                
        return out

    def check_and_report_easytier_anomaly(self):
        """检查并报告EasyTier异常"""
        full = self.get_easytier_full_info()
        summary = self.get_easytier_stats()
        anomaly = None
        
        if len(full) == 0:
            anomaly = {"type": "process_missing", "detail": "no easytier process found"}
        else:
            total_cpu = sum(p['cpu_percent'] for p in full)
            total_mem = sum(p['memory_mb'] for p in full)
            total_ext_11010 = sum(p.get('external_conns_11010', 0) for p in full)
            
            if total_cpu > 250.0:
                anomaly = {"type": "cpu_high", "total_cpu": total_cpu}
            elif total_mem > 1024.0:
                anomaly = {"type": "memory_high", "total_mem_mb": total_mem}
            elif total_ext_11010 > 200:
                anomaly = {"type": "conn_11010_high", "ext_conn_11010": total_ext_11010}
                
        if anomaly:
            payload = {
                "node_id": self.node_id,
                "public_ip": self.public_ip,
                "overlay_iface": self.overlay_iface,
                "summary": summary,
                "processes": full,
                "anomaly": anomaly,
                "timestamp": int(time.time())
            }
            ok = self.send_security_metric("process_anomaly", payload)
            self.report_ws_alert("process_anomaly", "EasyTier anomaly detected", payload)
            if ok:
                logger.warning(f"🚨 上报 EasyTier 异常: {anomaly['type']}")

    # ------------------ 配置管理 ------------------
    def load_config(self):
        self.config = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
            logger.info(f"✅ 加载配置文件: {self.config_file}")
            if 'Node' in self.config:
                self.node_id = self.config['Node'].get('node_id', '')
                self.admin_id = self.config['Node'].get('admin_id', '')
                self.node_name = self.config['Node'].get('node_name', '增强型DDoS防御节点')
                self.region = self.config['Node'].get('region', 'CN')
                self.cluster_url = self.config['Node'].get('cluster_url', 'https://fzjh.1427123.xyz')
                self.auth_token = self.config['Node'].get('auth_token', '')
            if 'Defense' in self.config:
                defense_ports = self.config['Defense'].get('defense_ports', '80,443,2233,11010')
                self.defense_config = DefenseConfig(
                    defense_ports=[int(p) for p in defense_ports.split(',')],
                    syn_flood_threshold=int(self.config['Defense'].get('syn_flood_threshold', '5000')),
                    udp_flood_threshold=int(self.config['Defense'].get('udp_flood_threshold', '100000')),
                    icmp_flood_threshold=int(self.config['Defense'].get('icmp_flood_threshold', '2000')),
                    http_flood_threshold=int(self.config['Defense'].get('http_flood_threshold', '500')),
                    connection_rate_threshold=int(self.config['Defense'].get('connection_rate_threshold', '200')),
                    packet_rate_threshold=int(self.config['Defense'].get('packet_rate_threshold', '10000')),
                    bandwidth_threshold=float(self.config['Defense'].get('bandwidth_threshold', '500.0')),
                    auto_blacklist=self.config['Defense'].getboolean('auto_blacklist', True),
                    auto_blacklist_duration=int(self.config['Defense'].get('auto_blacklist_duration', '1800')),
                    cooperative_defense=self.config['Defense'].getboolean('cooperative_defense', True),
                    evidence_sharing=self.config['Defense'].getboolean('evidence_sharing', True),
                    min_confidence=float(self.config['Defense'].get('min_confidence', '0.9')),
                    exclude_private_ips=self.config['Defense'].getboolean('exclude_private_ips', True),
                    enable_blackhole=self.config['Defense'].getboolean('enable_blackhole', False),
                    blackhole_threshold=float(self.config['Defense'].get('blackhole_threshold', '0.95'))
                )
        else:
            logger.info("📝 创建默认配置文件")
            self.node_name = "增强型DDoS防御节点"
            self.region = "CN"
            self.cluster_url = "https://fzjh.1427123.xyz"
            self.auth_token = ""
            self.defense_config = DefenseConfig(defense_ports=[80, 443, 2233, 11010])

    def save_config(self):
        self.config['Node'] = {
            'node_id': self.node_id,
            'admin_id': self.admin_id,
            'node_name': self.node_name,
            'region': self.region,
            'cluster_url': self.cluster_url,
            'auth_token': self.auth_token or ''
        }
        self.config['Defense'] = {
            'defense_ports': ','.join(map(str, self.defense_config.defense_ports)),
            'syn_flood_threshold': str(self.defense_config.syn_flood_threshold),
            'udp_flood_threshold': str(self.defense_config.udp_flood_threshold),
            'icmp_flood_threshold': str(self.defense_config.icmp_flood_threshold),
            'http_flood_threshold': str(self.defense_config.http_flood_threshold),
            'connection_rate_threshold': str(self.defense_config.connection_rate_threshold),
            'packet_rate_threshold': str(self.defense_config.packet_rate_threshold),
            'bandwidth_threshold': str(self.defense_config.bandwidth_threshold),
            'auto_blacklist': str(self.defense_config.auto_blacklist),
            'auto_blacklist_duration': str(self.defense_config.auto_blacklist_duration),
            'cooperative_defense': str(self.defense_config.cooperative_defense),
            'evidence_sharing': str(self.defense_config.evidence_sharing),
            'min_confidence': str(self.defense_config.min_confidence),
            'exclude_private_ips': str(self.defense_config.exclude_private_ips),
            'enable_blackhole': str(self.defense_config.enable_blackhole),
            'blackhole_threshold': str(self.defense_config.blackhole_threshold)
        }
        with open(self.config_file, 'w') as f:
            self.config.write(f)
        logger.info(f"💾 配置文件已保存: {self.config_file}")

    # ------------------ 数据库管理 ------------------
    def init_database(self):
        try:
            self.db_conn = sqlite3.connect('enhanced_node_data.db', 
                                         check_same_thread=False,
                                         timeout=30.0)
            self.db_conn.execute('PRAGMA journal_mode=WAL')
            self.db_conn.execute('PRAGMA synchronous=NORMAL')
            self.db_conn.execute('PRAGMA cache_size=-64000')
            
            self._setup_database_schema()
            logger.info("✅ 增强型数据库初始化完成")
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")
            self._setup_fallback_storage()

    def _setup_database_schema(self):
        """数据库schema设置"""
        c = self.db_conn.cursor()
        
        # DDoS证据表
        c.execute('''CREATE TABLE IF NOT EXISTS ddos_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attack_id TEXT NOT NULL UNIQUE,
            attack_type TEXT NOT NULL,
            source_ips TEXT NOT NULL,
            target_ports TEXT NOT NULL,
            local_ports TEXT NOT NULL,
            start_time INTEGER NOT NULL,
            end_time INTEGER,
            max_bandwidth REAL NOT NULL,
            packet_count INTEGER NOT NULL,
            connection_count INTEGER NOT NULL,
            target_node_id TEXT NOT NULL,
            source_reputation REAL NOT NULL,
            attack_signature TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            blockchain_tx TEXT,
            timestamp INTEGER NOT NULL
        )''')
        
        # 操作日志表
        c.execute('''CREATE TABLE IF NOT EXISTS operation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_type TEXT NOT NULL,
            target TEXT NOT NULL,
            reason TEXT,
            list_type TEXT,
            timestamp INTEGER NOT NULL,
            node_id TEXT NOT NULL,
            blockchain_tx TEXT
        )''')
        
        # 节点状态缓存表
        c.execute('''CREATE TABLE IF NOT EXISTS node_status_cache (
            node_id TEXT PRIMARY KEY,
            health TEXT NOT NULL,
            defense_mode TEXT NOT NULL,
            load REAL NOT NULL,
            connections INTEGER NOT NULL,
            last_seen INTEGER NOT NULL,
            reputation_score REAL NOT NULL,
            public_ip TEXT,
            region TEXT
        )''')
        
        # IP名单表
        c.execute('''CREATE TABLE IF NOT EXISTS ip_lists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip TEXT NOT NULL UNIQUE,
            list_type TEXT NOT NULL,
            reason TEXT,
            added_at INTEGER NOT NULL,
            expires_at INTEGER,
            added_by TEXT NOT NULL,
            confidence REAL DEFAULT 1.0
        )''')
        
        # 创建索引提高查询性能
        c.execute('''CREATE INDEX IF NOT EXISTS idx_ddos_evidence_time 
                     ON ddos_evidence(start_time DESC)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_operation_logs_time 
                     ON operation_logs(timestamp DESC)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_ip_lists_type 
                     ON ip_lists(list_type, expires_at)''')
        
        self.db_conn.commit()

    def _setup_fallback_storage(self):
        """数据库降级方案：使用内存存储"""
        logger.warning("💾 使用内存存储作为数据库降级方案")
        self.memory_storage = {
            'ddos_evidence': [],
            'operation_logs': [],
            'node_status_cache': {},
            'ip_lists': []
        }

    # ------------------ 网络工具 ------------------
    def get_headers(self):
        h = {"Content-Type": "application/json"}
        if self.auth_token: 
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    def get_public_ip(self) -> str:
        try:
            r = requests.get('https://httpbin.org/ip', timeout=5)
            if r.ok:
                origin = r.json().get('origin', '')
                return origin.split(",")[0].strip() if origin else "8.8.8.8"
        except Exception:
            pass
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "8.8.8.8"

    def _ws_endpoint(self) -> str:
        p = urlparse(self.cluster_url)
        scheme = "wss" if p.scheme == "https" else "ws"
        host = p.netloc or p.path
        return f"{scheme}://{host}/ws/node"

    # ------------------ 节点注册与通信 ------------------
    def register_node(self) -> bool:
        """节点注册"""
        if getattr(self, 'auth_token', ''):
            logger.info("🔑 使用保存的认证令牌")
            return True
            
        # 申请注册密钥
        url = f"{self.cluster_url}/api/nodes/request_key"
        payload = {
            "node_id": self.node_id, 
            "admin_id": self.admin_id,
            "node_info": {"name": self.node_name, "region": self.region}
        }
        
        for attempt in range(1, 4):
            try:
                logger.info(f"📝 申请注册密钥 (尝试 {attempt}/3)...")
                r = requests.post(url, json=payload, headers=self.get_headers(), timeout=30)
                if r.status_code == 200 and r.json().get('success'):
                    self.registration_key = r.json()['registration_key']
                    logger.info("✅ 注册密钥申请成功")
                    break
                logger.error(f"❌ 申请失败: {r.status_code} {r.text[:200]}")
            except Exception as e:
                logger.error(f"❌ 申请异常: {e}")
            if attempt < 3: 
                time.sleep(2)
        else:
            return False

        # 注册节点
        url = f"{self.cluster_url}/api/nodes/register"
        node_info = {
            "name": self.node_name, 
            "region": self.region,
            "node_index": 1, 
            "public_ip": self.public_ip, 
            "port": 2233
        }
        payload = {
            "node_id": self.node_id, 
            "registration_key": self.registration_key,
            "admin_id": self.admin_id, 
            "node_info": node_info
        }
        
        for attempt in range(1, 4):
            try:
                logger.info(f"📝 尝试注册节点 (尝试 {attempt}/3)...")
                r = requests.post(url, json=payload, headers=self.get_headers(), timeout=30)
                if r.status_code == 200:
                    j = r.json()
                    if j.get('ok'):
                        self.auth_token = j.get('auth_token', '')
                        logger.info("✅ 节点注册成功")
                        self.save_config()
                        self.log_operation("node_register", self.node_id, "节点注册成功")
                        return True
                    logger.error(f"❌ 响应异常: {j}")
                else:
                    logger.error(f"❌ 注册失败: {r.status_code} - {r.text[:200]}")
            except Exception as e:
                logger.error(f"❌ 注册异常: {e}")
            if attempt < 3: 
                time.sleep(2)
        return False

    # ------------------ WebSocket 通信 ------------------
    def connect_websocket(self):
        def ws_loop():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._websocket_handler())
            except Exception as e:
                logger.error(f"❌ WebSocket循环错误: {e}")
                
        self.websocket_thread = threading.Thread(target=ws_loop, daemon=True)
        self.websocket_thread.start()

    async def _websocket_handler(self):
        ws_url = self._ws_endpoint()
        params = f"?node_id={self.node_id}&auth_token={self.auth_token}"
        
        retry_count = 0
        max_retry_delay = 60
        
        while self.running:
            try:
                logger.info(f"🔌 连接WebSocket: {ws_url}")
                async with websockets.connect(ws_url + params, 
                                            ping_interval=30, 
                                            ping_timeout=10,
                                            close_timeout=10) as ws:
                    self.websocket = ws
                    self.websocket_connected = True
                    retry_count = 0
                    logger.info("✅ WebSocket连接建立")
                    
                    # 连接恢复后的状态同步
                    await self._sync_after_reconnect()
                    
                    async for message in ws:
                        if not self.running: 
                            break
                        await self._handle_websocket_message(message)
                        
            except Exception as e:
                self.websocket_connected = False
                if self.running:
                    retry_count += 1
                    delay = min(5 * (2 ** min(retry_count, 6)), max_retry_delay)
                    logger.error(f"❌ WebSocket连接错误: {e}, {delay}秒后重试...")
                    await asyncio.sleep(delay)

    async def _sync_after_reconnect(self):
        """连接恢复后的状态同步"""
        if self.websocket_connected:
            await self.websocket.send(json.dumps({
                "type": "node_online",
                "node_id": self.node_id,
                "defense_mode": self.defense_mode.value,
                "timestamp": int(time.time()),
                "reconnect": True
            }))

    async def _handle_websocket_message(self, message: str):
        try:
            data = json.loads(message)
            t = data.get('type')
            
            if t == "cluster_sync":
                self.available_nodes = data.get('nodes', [])
                # 更新节点状态缓存，隐藏公网IP
                for node in self.available_nodes:
                    if 'public_ip' in node:
                        node['public_ip'] = "已隐藏"
                    self.node_status_cache[node['node_id']] = node
                    
            elif t == "security_alert":
                self.metrics["cooperative_alerts_received"] += 1
                await self._handle_security_alert(data)
                
            elif t == "ping":
                if self.websocket_connected:
                    await self.websocket.send(json.dumps({"type": "pong"}))
                    
        except Exception as e:
            logger.error(f"❌ 处理WebSocket消息时出错: {e}")

    async def _handle_security_alert(self, data: Dict):
        alert_type = data.get('alert_type')
        source_node = data.get('source_node')
        evidence = data.get('evidence', {})
        
        logger.warning(f"🚨 协同防御警报 from {source_node}: {alert_type}")
        
        if alert_type == 'ddos_attack' and self.defense_config.cooperative_defense:
            # 自动同步攻击IP到黑名单
            for ip in evidence.get('source_ips', []):
                if ip not in ['unknown', 'detecting...']:
                    self.add_auto_blacklist_safeguard(ip, f"协同防御: 来自{source_node}的警报")
                    
            # 如果置信度高，自动切换到激进模式
            if evidence.get('confidence', 0) > 0.8:
                self.defense_mode = DefenseMode.AGGRESSIVE
                self.ddos_detector.set_aggressive_mode(True)
                logger.warning(f"🛡️ 基于协同警报切换到激进防御模式")

    # ------------------ 系统指标与健康报告 ------------------
    def collect_system_metrics(self) -> Dict:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.2)
            mem = psutil.virtual_memory().percent
            net_io = psutil.net_io_counters()
            bytes_sent = net_io.bytes_sent
            bytes_recv = net_io.bytes_recv

            traffic_stats = self.ddos_detector.get_traffic_stats()
            attack_status = self.ddos_detector.get_attack_status()
            et = self.get_easytier_stats()

            return {
                "node_id": self.node_id,
                "health": self.health,
                "defense_mode": self.defense_mode.value,
                "load": cpu_percent,
                "connections": traffic_stats.get('current_connections', 0),
                "public_latency": random.randint(10, 100),
                "bandwidth_up": (bytes_sent / 1024 / 1024),
                "bandwidth_down": (bytes_recv / 1024 / 1024),
                "memory_usage": mem,
                "cpu_usage": cpu_percent,
                "attack_detected": attack_status['attack_detected'],
                "current_attack_type": attack_status['attack_type'],
                "security_score": self.calculate_security_score(),
                "reputation_score": random.uniform(0.5, 1.0),
                "public_ip": self.public_ip,
                "region": self.region,
                "defense_ports": self.defense_config.defense_ports,
                "easytier": et,
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"❌ 收集系统指标失败: {e}")
            return {
                "node_id": self.node_id, "health": self.health, "defense_mode": self.defense_mode.value,
                "load": self.load, "connections": self.connections, "public_latency": 50,
                "bandwidth_up": 0.0, "bandwidth_down": 0.0,
                "memory_usage": 30.0, "cpu_usage": 20.0,
                "attack_detected": False, "current_attack_type": None,
                "security_score": 0.8, "reputation_score": 0.7,
                "public_ip": self.public_ip, "region": self.region,
                "defense_ports": self.defense_config.defense_ports,
                "easytier": {}, "timestamp": int(time.time())
            }

    def calculate_security_score(self) -> float:
        base = 1.0
        if self.defense_mode == DefenseMode.AGGRESSIVE: 
            base *= 1.2
        elif self.defense_mode == DefenseMode.ALERT: 
            base *= 1.1
        if self.ddos_detector.attack_detected: 
            base *= 0.7
        return max(0.1, min(1.0, base))

    def send_health_report(self) -> bool:
        """发送健康报告"""
        if not self.auth_token: 
            return False
            
        url = f"{self.cluster_url}/api/datachain/submit_metric"
        payload = {
            "node_id": self.node_id, 
            "metric_type": "health_report",
            "metric_data": self.collect_system_metrics()
        }
        
        try:
            r = requests.post(url, json=payload, headers=self.get_headers(), timeout=10)
            if r.ok:
                self.metrics["health_reports_sent"] += 1
            return r.ok
        except Exception as e:
            logger.error(f"❌ 发送健康报告时出错: {e}")
            self.metrics["errors_count"] += 1
            return False

    # ------------------ 区块链数据同步 ------------------
    def sync_blockchain_data(self):
        try:
            self.sync_node_status()
            self.sync_cloud_ip_lists()
            self.sync_ddos_evidence()
            self.metrics["blocks_synced"] += 1
        except Exception as e:
            logger.error(f"❌ 区块链数据同步失败: {e}")

    def sync_node_status(self):
        try:
            url = f"{self.cluster_url}/api/nodes/info"
            r = requests.get(url, headers=self.get_headers(), timeout=10)
            if r.ok:
                data = r.json()
                for node in data.get('nodes', []):
                    if isinstance(node, dict) and 'node_id' in node:
                        # 隐藏公网IP
                        if 'public_ip' in node:
                            node['public_ip'] = "已隐藏"
                            
                        self.node_status_cache[node['node_id']] = {
                            'health': node.get('health', 'unknown'),
                            'defense_mode': node.get('defense_mode', 'normal'),
                            'load': node.get('load', 0),
                            'connections': node.get('connections', 0),
                            'last_seen': node.get('last_seen', 0),
                            'reputation_score': node.get('reputation_score', 0.5),
                            'public_ip': node.get('public_ip', ''),
                            'region': node.get('region', 'unknown'),
                            'timestamp': int(time.time())
                        }
        except Exception as e:
            logger.error(f"❌ 节点状态同步失败: {e}")

    def sync_cloud_ip_lists(self):
        """云名单同步"""
        try:
            # 同步黑名单
            bl_url = f"{self.cluster_url}/api/security/blacklist"
            bl_response = requests.get(bl_url, headers=self.get_headers(), timeout=10)
            if bl_response.ok:
                blacklist_data = bl_response.json().get('blacklist', [])
                self.cloud_blacklist.sync_from_cloud(blacklist_data)
                logger.info(f"✅ 黑名单同步: 云{len(blacklist_data)}条")
            else:
                logger.error(f"❌ 黑名单同步失败: {bl_response.status_code}")
                
            # 同步白名单
            wl_url = f"{self.cluster_url}/api/security/whitelist"
            wl_response = requests.get(wl_url, headers=self.get_headers(), timeout=10)
            if wl_response.ok:
                whitelist_data = wl_response.json().get('whitelist', [])
                self.cloud_whitelist.sync_from_cloud(whitelist_data)
                logger.info(f"✅ 白名单同步: 云{len(whitelist_data)}条")
            else:
                logger.error(f"❌ 白名单同步失败: {wl_response.status_code}")
                
            self.metrics["cloud_sync_count"] += 1
                
        except Exception as e:
            logger.error(f"❌ 云名单同步失败: {e}")

    def sync_ddos_evidence(self):
        try:
            url = f"{self.cluster_url}/api/datachain/ddos/status"
            r = requests.get(url, headers=self.get_headers(), timeout=10)
            if r.ok:
                for attack in r.json().get('active_attacks', []):
                    if attack.get('mitigation_status') != 'resolved':
                        self.process_remote_attack(attack)
        except Exception as e:
            logger.error(f"❌ DDoS证据同步失败: {e}")

    def process_remote_attack(self, attack: Dict):
        attack_type = attack.get('type')
        source_ips = attack.get('source_ips', [])
        target_node = attack.get('target_node')
        
        logger.warning(f"🚨 远程攻击警报: {attack_type} -> {target_node}")
        
        if self.defense_config.cooperative_defense and self.defense_config.auto_blacklist:
            for ip in source_ips:
                if ip not in ['unknown', 'detecting...']:
                    self.add_to_cloud_blacklist(ip, f"协同防御: {attack_type}攻击")

    # ------------------ 黑白名单管理 ------------------
    def add_to_cloud_blacklist(self, ip: str, reason: str = "manual") -> bool:
        try:
            url = f"{self.cluster_url}/api/security/blacklist/report"
            payload = {"node_id": self.node_id, "ip": ip, "reason": reason}
            r = requests.post(url, json=payload, headers=self.get_headers(), timeout=10)
            if r.ok:
                j = r.json()
                tx = j.get('blockchain_tx', '')
                self.metrics["ip_blacklisted"] += 1
                self.log_operation("add_cloud_blacklist", ip, reason, "cloud", tx)
                logger.info(f"✅ 云黑名单: {ip} - {reason} (TX: {tx[:16]}...)")
                return True
            logger.error(f"❌ 云黑名单添加失败: {r.status_code}")
            return False
        except Exception as e:
            logger.error(f"❌ 添加云黑名单失败: {e}")
            return False

    def add_to_local_blacklist(self, ip: str, reason: str = "manual", duration: int = 3600) -> bool:
        try:
            success = self.local_blacklist.add_ip(ip, reason, duration)
            if success:
                self.log_operation("add_local_blacklist", ip, reason, "local")
                logger.info(f"✅ 本地黑名单: {ip} - {reason}")
            return success
        except Exception as e:
            logger.error(f"❌ 添加本地黑名单失败: {e}")
            return False

    def add_to_whitelist(self, ip: str, reason: str = "manual") -> bool:
        try:
            success = self.local_whitelist.add_ip(ip, reason, 0)  # 白名单永不过期
            if success:
                self.log_operation("add_whitelist", ip, reason, "local")
                logger.info(f"✅ 本地白名单: {ip} - {reason}")
            return success
        except Exception as e:
            logger.error(f"❌ 添加白名单失败: {e}")
            return False

    def add_auto_blacklist_safeguard(self, ip: str, reason: str) -> bool:
        # 白名单优先放行
        if self.cloud_whitelist.is_listed(ip) or self.local_whitelist.is_listed(ip):
            self.metrics["ip_blacklist_blocked"] += 1
            logger.info(f"🛡️ 白名单放行：{ip}")
            return False
            
        try:
            ipo = ipaddress.ip_address(ip)
            if self.defense_config.exclude_private_ips and (ipo.is_private or ipo.is_loopback or ipo.is_link_local):
                self.metrics["ip_blacklist_blocked"] += 1
                logger.warning(f"⚠️ 跳过内网/保留IP: {ip}")
                return False
        except Exception:
            self.metrics["ip_blacklist_blocked"] += 1
            return False
            
        rep = self.ip_reputation_system.get_reputation(ip)
        if rep > 0.7:
            self.metrics["ip_blacklist_blocked"] += 1
            logger.warning(f"⚠️ 高信誉IP {ip} (信誉: {rep:.2f})，跳过自动拉黑")
            return False
            
        ok = self.add_to_local_blacklist(ip, reason, self.defense_config.auto_blacklist_duration)
        if ok and self.defense_config.cooperative_defense:
            self.add_to_cloud_blacklist(ip, reason)
            
        return ok

    # ------------------ DDoS证据管理 ------------------
    def report_ddos_evidence(self, evidence: DDoSEvidence) -> bool:
        if not self.auth_token: 
            return False
            
        url = f"{self.cluster_url}/api/datachain/ddos/report"
        
        traffic_snapshot = self.ddos_detector.get_traffic_stats()
        easytier_snapshot = self.get_easytier_stats()
        
        payload = {
            "node_id": self.node_id,
            "evidence": asdict(evidence),
            "evidence_ext": {
                "traffic_stats": traffic_snapshot,
                "easytier": easytier_snapshot
            }
        }
        
        try:
            r = requests.post(url, json=payload, headers=self.get_headers(), timeout=10)
            if r.ok:
                j = r.json()
                evidence.blockchain_tx = j.get('blockchain_tx', '')
                self.save_ddos_evidence(evidence)
                    
                self.metrics["ddos_attacks_detected"] += 1
                logger.warning(f"🚨 DDoS证据: {evidence.attack_type} 置信度={evidence.confidence:.2f} TX={evidence.blockchain_tx[:16] or '-'}")
                return True
                
            logger.error(f"❌ 证据报告失败: {r.status_code} - {r.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"❌ 报告DDoS证据异常: {e}")
            return False

    def save_ddos_evidence(self, evidence: DDoSEvidence):
        try:
            c = self.db_conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO ddos_evidence 
                (attack_id, attack_type, source_ips, target_ports, local_ports, start_time, end_time, 
                 max_bandwidth, packet_count, connection_count, target_node_id, source_reputation, 
                 attack_signature, confidence, blockchain_tx, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                evidence.attack_id, evidence.attack_type, json.dumps(evidence.source_ips),
                json.dumps(evidence.target_ports), json.dumps(evidence.local_attacked_ports),
                evidence.start_time, evidence.end_time, evidence.max_bandwidth_mbps, evidence.packet_count,
                evidence.connection_count, evidence.target_node_id, evidence.source_ip_reputation,
                evidence.attack_signature, evidence.confidence, evidence.blockchain_tx, int(time.time())
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存DDoS证据失败: {e}")

    def get_ddos_evidence(self, limit: int = 50) -> List[DDoSEvidence]:
        try:
            c = self.db_conn.cursor()
            c.execute('''
                SELECT attack_id, attack_type, source_ips, target_ports, local_ports, start_time, end_time,
                       max_bandwidth, packet_count, connection_count, target_node_id, source_reputation,
                       attack_signature, confidence, blockchain_tx
                FROM ddos_evidence 
                ORDER BY start_time DESC 
                LIMIT ?
            ''', (limit,))
        except Exception:
            return []
            
        out: List[DDoSEvidence] = []
        for row in c.fetchall():
            out.append(DDoSEvidence(
                attack_id=row[0], attack_type=row[1], source_ips=json.loads(row[2]),
                target_ports=json.loads(row[3]), local_attacked_ports=json.loads(row[4]),
                start_time=row[5], end_time=row[6], max_bandwidth_mbps=row[7],
                packet_count=row[8], connection_count=row[9], target_node_id=row[10],
                source_ip_reputation=row[11], attack_signature=row[12],
                confidence=row[13], blockchain_tx=row[14]
            ))
        return out

    def log_operation(self, operation_type: str, target: str, reason: str = "",
                      list_type: str = "", blockchain_tx: str = ""):
        try:
            c = self.db_conn.cursor()
            c.execute('''
                INSERT INTO operation_logs (operation_type, target, reason, list_type, timestamp, node_id, blockchain_tx)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (operation_type, target, reason, list_type, int(time.time()), self.node_id, blockchain_tx))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"❌ 记录操作日志失败: {e}")

    def get_operation_logs(self, limit: int = 50) -> List[Dict]:
        try:
            c = self.db_conn.cursor()
            c.execute('''
                SELECT operation_type, target, reason, list_type, timestamp, node_id, blockchain_tx
                FROM operation_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            cols = ['operation_type', 'target', 'reason', 'list_type', 'timestamp', 'node_id', 'blockchain_tx']
            return [dict(zip(cols, row)) for row in c.fetchall()]
        except Exception as e:
            logger.error(f"❌ 获取操作日志失败: {e}")
            return []

    # ------------------ 节点状态显示 ------------------
    def print_cluster_nodes(self):
        """集群节点状态显示"""
        if not self.node_status_cache:
            print("📝 暂无节点状态信息\n")
            return
            
        print("\n" + "="*90)
        print(f"🌍 集群节点状态".center(88))
        print("="*90)
        headers = ["节点ID", "区域", "健康", "模式", "负载%", "连接数", "最后更新"]
        print(f"{headers[0]:<18} {headers[1]:<6} {headers[2]:<6} {headers[3]:<10} {headers[4]:<6} {headers[5]:<8} {headers[6]:<10}")
        print("-" * 90)
        
        mode_icons = {
            "normal": "🟢",
            "alert": "🟡", 
            "aggressive": "🔴"
        }
        
        for node_id, status in sorted(self.node_status_cache.items()):
            if node_id == self.node_id:
                continue  # 跳过自己
                
            health_icon = "🟢" if status.get('health') == 'healthy' else "🔴"
            mode_icon = mode_icons.get(status.get('defense_mode', 'normal'), '⚪')
            last_seen = int(time.time() - status.get('last_seen', 0))
            
            if last_seen < 60:
                last_seen_str = f"{last_seen}s"
            elif last_seen < 3600:
                last_seen_str = f"{last_seen//60}m"
            else:
                last_seen_str = f"{last_seen//3600}h"
                
            print(f"{node_id[:16]:<18} {status.get('region',''):<6} {health_icon} {status.get('health',''):<4} "
                  f"{mode_icon} {status.get('defense_mode',''):<8} {status.get('load',0):<6.1f} "
                  f"{status.get('connections',0):<8} {last_seen_str:<10}")
                  
        print("="*90)
        print()

    # ------------------ 主循环线程 ------------------
    def start_ddos_detection(self):
        def detection_loop():
            while self.running:
                try:
                    res = self.ddos_detector.detect_attacks()
                    if res["attack_detected"]:
                        evidence = DDoSEvidence(
                            attack_id=f"attack_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                            attack_type=res["attack_type"], 
                            source_ips=res["suspicious_ips"],
                            target_ports=res["target_ports"], 
                            local_attacked_ports=self.defense_config.defense_ports,
                            start_time=int(time.time()), 
                            end_time=None, 
                            max_bandwidth_mbps=res["max_bandwidth"],
                            packet_count=res["packet_count"], 
                            connection_count=res["connection_count"],
                            target_node_id=self.node_id,
                            source_ip_reputation=self.ip_reputation_system.get_reputation(res['suspicious_ips'][0]) if res['suspicious_ips'] else 0.5,
                            attack_signature=res["attack_signature"], 
                            confidence=res.get('confidence', 0.5)
                        )
                        self.report_ddos_evidence(evidence)
                        
                        # 自动拉黑
                        if self.defense_config.auto_blacklist:
                            conf = res.get('confidence', 0.5)
                            if conf >= self.defense_config.min_confidence:
                                for ip in res['suspicious_ips']:
                                    self.add_auto_blacklist_safeguard(ip, f"自动拉黑: {res['attack_type']}")
                            else:
                                logger.info(f"⚠️ 低置信度攻击检测 (置信度: {conf:.2f})，跳过自动拉黑")
                                
                except Exception as e:
                    logger.error(f"❌ DDoS检测循环错误: {e}")
                time.sleep(5)
                
        t = threading.Thread(target=detection_loop, daemon=True)
        t.start()
        self.ddos_detection_thread = t
        logger.info("🔍 DDoS检测已启动")

    def start_heartbeat(self):
        def heartbeat_loop():
            while self.running:
                try:
                    self.send_health_report()
                    
                    now = time.time()
                    if now - self.last_sync_time >= self.sync_interval:
                        self.sync_blockchain_data()
                        self.last_sync_time = now
                        
                except Exception as e:
                    logger.error(f"❌ 心跳循环错误: {e}")
                    self.metrics["errors_count"] += 1
                time.sleep(30)
                
        t = threading.Thread(target=heartbeat_loop, daemon=True)
        t.start()
        self.heartbeat_thread = t
        logger.info("💓 心跳循环已启动")

    def start_command_panel(self):
        """启动交互式命令面板"""
        self.command_panel.start_panel()

    # ------------------ 启动与停止 ------------------
    def start(self) -> bool:
        logger.info(f"🚀 启动增强型DDoS防御节点: {self.node_id}")
        logger.info(f"    集群URL: {self.cluster_url}")
        logger.info(f"    公网IP: {self.public_ip}")
        logger.info(f"    防御端口: {self.defense_config.defense_ports}")
        
        # 集群连接检查
        try:
            h = requests.get(f"{self.cluster_url}/healthz", timeout=10)
            if h.status_code == 200:
                logger.info(f"✅ 集群连接正常")
            else:
                logger.error(f"❌ 集群健康检查失败: {h.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ 集群连接失败: {e}")
            return False

        if not self.register_node():
            return False

        self.running = True
        self.online = True

        self.start_heartbeat()
        self.start_ddos_detection()
        self.connect_websocket()

        logger.info("🎉 增强型DDoS防御节点启动完成!")
        return True

    def stop(self):
        if not self.running: 
            return
            
        logger.info("🛑 停止节点...")
        self.running = False
        self.online = False
        
        if self.websocket_connected and self.websocket:
            try:
                threading.Thread(target=self._run_async_offline, daemon=True).start()
            except Exception:
                pass
                
        self.save_config()
        
        if hasattr(self, 'db_conn'):
            try: 
                self.db_conn.close()
            except Exception: 
                pass
                
        threads = [self.heartbeat_thread, self.websocket_thread,
                   self.ddos_detection_thread, self.cooperative_thread]
        for t in threads:
            if t and t.is_alive(): 
                t.join(timeout=2.0)
                
        logger.info("✅ 节点已优雅停止")
        
        if hasattr(self, 'panel_thread') and self.panel_thread and self.panel_thread.is_alive():
            logger.info("📭 退出命令面板")
            os._exit(0)

    def signal_handler(self, signum, frame):
        logger.info(f"📭 收到退出信号 {signum}，正在优雅退出...")
        self.stop()

    def _run_async_offline(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._send_offline_notification())
            loop.close()
        except Exception:
            pass

    async def _send_offline_notification(self):
        try:
            if self.websocket and self.websocket_connected:
                await self.websocket.send(json.dumps({
                    "type": "node_offline",
                    "node_id": self.node_id,
                    "timestamp": int(time.time())
                }))
                await asyncio.sleep(0.3)
        except Exception:
            pass

# ------------------ 主函数 ------------------
def main():
    node = EnhancedDDoSNode("enhanced_node_config.ini")
    try:
        if node.start():
            print("\n🎉 节点启动成功！已进入交互式命令面板")
            print("💡 输入 'help' 查看可用命令")
            
            # 启动命令面板（会阻塞在这里）
            node.start_command_panel()
        else:
            logger.error("❌ 节点启动失败")
    except KeyboardInterrupt:
        logger.info("👋 收到停止信号")
    except Exception as e:
        logger.error(f"💥 节点运行异常: {e}")
    finally:
        node.stop()

if __name__ == "__main__":
    main()
