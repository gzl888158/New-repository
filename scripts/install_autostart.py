"""
安装开机自启动脚本
将交易系统添加到Windows启动项
"""
import os
import sys
import shutil
from pathlib import Path

def install_autostart():
    """安装开机自启动"""
    project_dir = Path(__file__).parent.resolve()
    startup_dir = Path(os.environ.get('APPDATA', '')) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs' / 'Startup'
    
    if not startup_dir.exists():
        startup_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建启动快捷方式
    bat_source = project_dir / 'start_radical.bat'
    shortcut_name = 'OKX_Trading_System.bat'
    shortcut_path = startup_dir / shortcut_name
    
    try:
        shutil.copy2(str(bat_source), str(shortcut_path))
        print(f"[OK] 开机自启动已安装")
        print(f"     快捷方式路径: {shortcut_path}")
        print(f"     启动脚本: {bat_source}")
    except Exception as e:
        print(f"[FAIL] 安装失败: {e}")
        return False
    
    # 验证
    if shortcut_path.exists():
        print(f"[OK] 验证通过 - 启动项已存在")
        return True
    else:
        print(f"[FAIL] 验证失败 - 未找到启动项")
        return False

def uninstall_autostart():
    """卸载开机自启动"""
    startup_dir = Path(os.environ.get('APPDATA', '')) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs' / 'Startup'
    shortcut_path = startup_dir / 'OKX_Trading_System.bat'
    
    if shortcut_path.exists():
        try:
            shortcut_path.unlink()
            print(f"[OK] 开机自启动已卸载")
            return True
        except Exception as e:
            print(f"[FAIL] 卸载失败: {e}")
            return False
    else:
        print(f"[INFO] 未找到开机自启动项")
        return True

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'uninstall':
        uninstall_autostart()
    else:
        install_autostart()
