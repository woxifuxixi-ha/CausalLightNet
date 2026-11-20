"""
配置文件读取器 - 支持SEVIR配置
支持行内注释（#）
"""

def read_config(config_file='config_train_sevir'):
    """
    读取配置文件
    
    参数:
        config_file: 配置文件路径，默认为'config_train_sevir'
    
    返回:
        config_info: 配置字典
    """
    ConfigFilePath = config_file
    config_info = {}
    
    try:
        with open(ConfigFilePath) as file:
            for line in file:
                line = line.rstrip('\n')
                line = line.rstrip('\r\n')
                
                # 跳过空行和注释行
                if not line or line.strip().startswith('#'):
                    continue
                
                # 解析配置项
                if '=' not in line:
                    continue
                
                item = line.split('=')
                key = item[0].strip()
                value = item[1].strip() if len(item) > 1 else ''
                
                # 关键修复：去掉行内注释
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                # 根据键名解析值类型
                if key in ['WRFFileDir', 'TruthFileDir', 'TruthFileDirGrid', 
                          'VisResultFileDir', 'GraphFileDir', 'ModelFileDir', 
                          'RecordFileDir', 'NetName', 'Device', 
                          'ScanStartTime', 'ScanEndTime']:
                    config_info[key] = value
                    
                elif key in ['GridRowColNum', 'ForecastHourNum', 'TruthHistoryHourNum',
                           'Batchsize', 'EpochNum', 'WRFChannelNum']:
                    try:
                        config_info[key] = int(value)
                    except ValueError:
                        print(f"Warning: Cannot convert '{key}' value '{value}' to int, skipping")
                        continue
                    
                elif key in ['LearningRate']:
                    try:
                        config_info[key] = float(value)
                    except ValueError:
                        print(f"Warning: Cannot convert '{key}' value '{value}' to float, skipping")
                        continue
                    
                elif key == 'GraphChooseLayers':
                    l_list = []
                    for c in value.split(','):
                        c_clean = c.strip()
                        if c_clean:
                            try:
                                l_list.append(int(c_clean))
                            except ValueError:
                                print(f"Warning: Cannot convert '{c_clean}' to int in GraphChooseLayers")
                    config_info[key] = l_list
                    
                else:
                    # 未知的配置项，作为字符串保存
                    print(f'Warning: Unknown config item: {key} = {value}')
                    config_info[key] = value
                    
    except FileNotFoundError:
        print(f"Error: Configuration file '{ConfigFilePath}' not found!")
        print("Please create a configuration file or check the path.")
        raise
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return config_info


if __name__ == "__main__":
    # 测试配置读取
    try:
        config = read_config('config_train_sevir')
        print("Configuration loaded successfully:")
        print("-" * 60)
        for key, value in config.items():
            print(f"{key:25s} = {value}")
        print("-" * 60)
        print(f"Total {len(config)} configuration items")
    except Exception as e:
        print(f"Failed to load configuration: {e}")