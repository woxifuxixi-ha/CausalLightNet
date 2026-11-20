"""
因果先验解析器
从因果发现报告中提取因果关系并转换为可用的先验权重
"""
import numpy as np
import re
from typing import Dict, List, Tuple


class CausalPriorParser:
    """解析因果发现报告，提取因果关系作为先验知识"""
    
    def __init__(self, report_path: str):
        """
        参数:
            report_path: 因果报告文件路径
        """
        self.report_path = report_path
        self.causal_graph = {}
        self.strength_matrix = {}
        self.delay_matrix = {}
        
        self._parse_report()
        self._normalize_strengths()
    
    def _parse_report(self):
        """解析因果报告文件"""
        with open(self.report_path, 'r') as f:
            content = f.read()
        
        # 提取详细因果关系部分
        pattern = r'Target: (\w+)\n((?:  <- .*\n)*)'
        matches = re.findall(pattern, content)
        
        for target, relationships in matches:
            self.causal_graph[target] = []
            
            # 解析每条因果关系
            rel_pattern = r'<- (\w+)\s+\(delay=\s*(\d+).*strength=([\d.e+-]+)\)'
            rels = re.findall(rel_pattern, relationships)
            
            for source, delay, strength in rels:
                self.causal_graph[target].append({
                    'source': source,
                    'delay': int(delay),
                    'strength': float(strength)
                })
        
        print("因果关系解析完成:")
        for target, sources in self.causal_graph.items():
            print(f"  {target}: {len(sources)} 个因果源")
    
    def _normalize_strengths(self):
        """归一化因果强度到 [0, 1] 范围"""
        # 收集所有强度值
        all_strengths = []
        for target, sources in self.causal_graph.items():
            for rel in sources:
                all_strengths.append(rel['strength'])
        
        if not all_strengths:
            return
        
        # 使用对数尺度归一化（因为强度值范围很大）
        log_strengths = np.log10(np.array(all_strengths) + 1e-10)
        min_log = log_strengths.min()
        max_log = log_strengths.max()
        
        # 构建归一化的强度矩阵
        for target, sources in self.causal_graph.items():
            if target not in self.strength_matrix:
                self.strength_matrix[target] = {}
                self.delay_matrix[target] = {}
            
            for rel in sources:
                source = rel['source']
                log_strength = np.log10(rel['strength'] + 1e-10)
                
                # 归一化到 [0.1, 1.0] 范围（保持最小权重为0.1）
                normalized = 0.1 + 0.9 * (log_strength - min_log) / (max_log - min_log + 1e-10)
                
                self.strength_matrix[target][source] = normalized
                self.delay_matrix[target][source] = rel['delay']
    
    def get_causal_weights(self, target: str) -> Dict[str, float]:
        """
        获取目标变量的因果权重
        
        参数:
            target: 目标变量名称（如 'sevir_lght'）
        
        返回:
            因果权重字典 {source: weight}
        """
        return self.strength_matrix.get(target, {})
    
    def get_causal_delays(self, target: str) -> Dict[str, int]:
        """
        获取目标变量的因果延迟
        
        参数:
            target: 目标变量名称
        
        返回:
            因果延迟字典 {source: delay_steps}
        """
        return self.delay_matrix.get(target, {})
    
    def print_summary(self):
        """打印因果先验摘要"""
        print("\n" + "="*70)
        print("因果先验摘要")
        print("="*70)
        
        for target in sorted(self.strength_matrix.keys()):
            print(f"\n目标: {target}")
            weights = self.strength_matrix[target]
            delays = self.delay_matrix[target]
            
            # 按权重排序
            sorted_sources = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            
            for source, weight in sorted_sources:
                delay = delays[source]
                print(f"  <- {source:15s} | 权重: {weight:.4f} | 延迟: {delay} 步")
    
    def get_lightning_causal_config(self) -> Dict:
        """
        为闪电预测获取因果配置
        
        返回:
            因果配置字典，包含权重和延迟信息
        """
        target = 'sevir_lght'
        
        if target not in self.strength_matrix:
            print(f"警告: 未找到 {target} 的因果关系")
            return {}
        
        config = {
            'weights': self.strength_matrix[target],
            'delays': self.delay_matrix[target],
            'sources': list(self.strength_matrix[target].keys())
        }
        
        print(f"\n闪电预测因果配置:")
        print(f"  因果源数量: {len(config['sources'])}")
        print(f"  因果源: {config['sources']}")
        
        return config


if __name__ == "__main__":
    # 测试解析器
    report_path = "causal_report_strict.txt"
    
    try:
        parser = CausalPriorParser(report_path)
        parser.print_summary()
        
        # 获取闪电预测的因果配置
        print("\n" + "="*70)
        lght_config = parser.get_lightning_causal_config()
        
        print("\n闪电预测因果权重:")
        for source, weight in lght_config['weights'].items():
            print(f"  {source}: {weight:.4f}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {report_path}")
        print("请确保因果报告文件在当前目录下")