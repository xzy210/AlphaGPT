"""测试 xtquant 数据获取"""
from xtquant import xtdata

# 测试股票代码
test_code = "000001.SZ"

print("=" * 50)
print("测试 XtQuant 数据获取")
print("=" * 50)

# 1. 先下载数据
print(f"\n1. 下载 {test_code} 的历史数据...")
xtdata.download_history_data(test_code, '1d', '', '')

# 2. 获取数据并检查格式
print(f"\n2. 获取 {test_code} 的K线数据...")
data = xtdata.get_market_data(
    field_list=['time', 'open', 'high', 'low', 'close', 'volume', 'amount'],
    stock_list=[test_code],
    period='1d',
    start_time='',
    end_time='',
    count=10,  # 只获取10条
    dividend_type='front',
    fill_data=True
)

print(f"\n3. 数据类型: {type(data)}")

if data is None:
    print("   数据为 None!")
elif isinstance(data, dict):
    print(f"   字典的键: {data.keys()}")
    
    if 'close' in data:
        print(f"\n   data['close'] 类型: {type(data['close'])}")
        
        if isinstance(data['close'], dict):
            print(f"   data['close'] 的键: {data['close'].keys()}")
            if test_code in data['close']:
                close_data = data['close'][test_code]
                print(f"   data['close']['{test_code}'] 类型: {type(close_data)}")
                print(f"   data['close']['{test_code}'] 长度: {len(close_data)}")
                print(f"   前5个值: {close_data[:5] if len(close_data) > 5 else close_data}")
        else:
            print(f"   data['close'] 内容: {data['close'][:5] if len(data['close']) > 5 else data['close']}")
    
    if 'time' in data:
        print(f"\n   data['time'] 类型: {type(data['time'])}")
        if isinstance(data['time'], dict) and test_code in data['time']:
            time_data = data['time'][test_code]
            print(f"   时间数据前5个: {time_data[:5] if len(time_data) > 5 else time_data}")
else:
    print(f"   意外的数据类型: {type(data)}")
    print(f"   数据内容: {data}")

# 3. 尝试另一种获取方式
print("\n" + "=" * 50)
print("尝试 get_local_data 方法...")
try:
    data2 = xtdata.get_local_data(
        field_list=['time', 'open', 'high', 'low', 'close', 'volume', 'amount'],
        stock_list=[test_code],
        period='1d',
        count=10
    )
    print(f"get_local_data 返回类型: {type(data2)}")
    if data2:
        print(f"键: {data2.keys() if isinstance(data2, dict) else 'N/A'}")
except Exception as e:
    print(f"get_local_data 失败: {e}")

print("\n" + "=" * 50)
print("测试完成")

