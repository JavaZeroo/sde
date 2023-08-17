import torch
import plotly.graph_objects as go
import numpy as np
from rich.progress import track
import imageio
from matplotlib import pyplot as plt
from pathlib import Path
import shutil
import matplotlib.animation as animation
from rich import print
from tqdm import tqdm
import torchvision

def draw_gaussian2d(ts, bridge, colors=True):
    def get_color(point):
        x, y = point
        if x > 0 and y > 0:
            return 0
        elif x > 0 and y < 0:
            return 1
        elif x < 0 and y > 0:
            return 2
        else:
            return 3

    # 生成数据
    color_list = ['#37306B', '#66347F', '#9E4784', '#D27685']  # 对应于目标均值的颜色

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 为每个样本绘制bridge
    for i in range(bridge.shape[1]):
        x = bridge[:, i, 0].numpy()  # X坐标
        y = bridge[:, i, 1].numpy()  # Y坐标
        z = ts.numpy()               # 时间作为Z坐标
        c = color_list[get_color((x[-1], y[-1]))] if colors else '#D27685'
        ax.plot(x, y, z, color=c, label=f'Sample {i+1}', alpha=0.3)

    # 添加标签和图例
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time')

    fig.show()
    
def draw_gaussian2d_gif(bridge):
    """
    Decrapated
    """
    def get_color(point):
        x, y = point
        if x > 0 and y > 0:
            return 0
        elif x > 0 and y < 0:
            return 1
        elif x < 0 and y > 0:
            return 2
        else:
            return 3
    colors = ['b', 'g', 'r', 'c']  # 对应于目标均值的颜色
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    # 定义更新函数
    def update(frame):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        for i in range(bridge.shape[1]):
            x = bridge[:frame + 1, i, 0].numpy()  # 注意：frame + 1
            y = bridge[:frame + 1, i, 1].numpy()  # 注意：frame + 1
            if x.size > 0 and y.size > 0:  # 确保x和y不为空
                color_index = get_color((x[-1], y[-1]))
                ax.plot(x, y, color=colors[color_index], alpha=0.5)
                ax.scatter(x[-1], y[-1], color=colors[color_index], alpha=1, s=10)
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(ts), interval=100)
    # 保存为GIF需要花费6分钟左右
    ani.save(log_dir / 'brownian_bridge.gif', writer='imagemagick')


def print_debug(*args):
    print('='*20)
    for arg in args:
        print(arg.shape, arg.dtype, arg.device)
        
        
# # 导入Plotly库
import plotly.graph_objects as go

def draw_3d_line(bridge_tensor, sample_rate=0.1, time_rate=0.1, colors=True):
    assert sample_rate > 0 and sample_rate <= 1 and time_rate > 0 and time_rate <= 1, "sample_rate and time_rate must be in (0, 1]"
    if sample_rate <= 1:
        bridge_tensor = bridge_tensor[:, ::int(1/sample_rate), :]
    if time_rate <= 1:
        bridge_tensor = bridge_tensor[::int(1/time_rate), :, :]
    
    def get_color(point):
        x, y = point
        if x > 0 and y > 0:
            return 0
        elif x > 0 and y < 0:
            return 1
        elif x < 0 and y > 0:
            return 2
        else:
            return 3
        
    color_list = ['#F9ED69', '#F08A5D', '#B83B5E', '#6A2C70']  # 对应于目标均值的颜色
    
    # 创建一个存储线条的列表
    lines = []
    x = bridge_tensor[:, :, 0].numpy()
    y = bridge_tensor[:, :, 1].numpy()
    z = torch.arange(bridge_tensor.size(0)).numpy()
    # 循环添加每一条线
    for i in range(bridge_tensor.size(1)):
        c = color_list[get_color((x[-1, i], y[-1, i]))] if colors else '#B83B5E'
        line = go.Scatter3d(
            z=z, y=y[:, i], x=x[:, i],
        marker=dict(
            size=2,
            color=c,
            colorscale='Viridis',
            opacity=0.5
        ),
        line=dict(
            width=2, 
            color=c,
        )
        )
        lines.append(line)

    # 创建3D图形
    fig = go.Figure(data=lines)

    # 设置轴标签
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Time (Z)',
        ),
        margin=dict(l=1, r=1, t=1, b=1),
        paper_bgcolor="#efefef",
    )

    # 显示图形
    fig.show()
    
    
def plot_source_and_target(sour, targ, left_title="Source Sample", right_title="Target Sample", save_path=None, bound=8):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(*sour.T, c='#D27685', s=10, alpha=0.6)
    axs[1].scatter(*targ.T, c='#37306B', s=10, alpha=0.6)
    axs[0].set_title(left_title)
    axs[1].set_title(right_title)
    axs[0].grid(True, alpha=0.3)
    axs[1].grid(True, alpha=0.3)
    axs[0].set_xlim(-bound, bound)
    axs[0].set_ylim(-bound, bound)
    axs[1].set_xlim(-bound, bound)
    axs[1].set_ylim(-bound, bound)
    fig.show()
    if save_path is not None:
        fig.savefig(save_path)
    
def save_gif_frame(bridge, save_path=None, bound=10):
    assert save_path is not None, "save_path cannot be None"
    save_path = Path(save_path)
    bridge = bridge[::10, :, :].numpy()  # 降低采样率

    temp_dir = save_path / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    frame = 0
    
    color_map = -np.sqrt(bridge[0, :, 0]**2 + bridge[0, :, 1]**2)
    for i in track(range(bridge.shape[0]), description="Processing image"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        x = bridge[i, :, 0]  # 注意：
        y = bridge[i, :, 1]  # 注意：
        
        ax.scatter(x, y, c=color_map, alpha=1, s=10)
        fig.savefig(save_path / 'temp' / f'{frame:03d}.png', dpi=100)
        frame += 1
        fig.show()
        plt.close('all')
    frames = []
    for i in range(bridge.shape[0]):
        frame_image = imageio.imread(save_path / 'temp' / f'{i:03d}.png')
        frames.append(frame_image)
    imageio.mimsave(save_path / 'brownian_bridge.gif', frames, duration=0.2)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

def binary(x):
    x = (x - x.min()) / (x.max() - x.min())
    x = torch.where(x > 0.3, torch.ones_like(x), torch.zeros_like(x))
    return x

def concat_mnist(data, x=5, y=5):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.view(x, y, 28, 28)
    data = torch.concatenate([torch.cat([data[i, j] for j in range(x)], dim=1) for i in range(y)], dim=0)
    return data

def plot_source_and_target_mnist(sour, targ, left_title="Source Sample", right_title="Target Sample", save_path=None, show=False):
    sour = concat_mnist(sour)
    targ = concat_mnist(targ)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(sour.cpu().numpy(), cmap='gray')
    axs[1].imshow(targ.cpu().numpy(), cmap='gray')
    axs[0].set_title(left_title)
    axs[1].set_title(right_title)
    if show:
        fig.show()
    if save_path is not None:
        fig.savefig(save_path)

def save_gif_frame_mnist(bridge, save_path=None, save_name='brownian_bridge.gif'):
    assert save_path is not None, "save_path cannot be None"
    save_path = Path(save_path)
    bridge = bridge[::10, :, :].numpy()  # 降低采样率

    temp_dir = save_path / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    frame = 0
    
    # color_map = -np.sqrt(bridge[0, :, 0]**2 + bridge[0, :, 1]**2)
    for i in track(range(bridge.shape[0]), description="Processing image"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.clear()
        ax.imshow(concat_mnist(bridge[i]).cpu().numpy(), cmap='gray')
        fig.savefig(save_path / 'temp' / f'{frame:03d}.png', dpi=100)
        frame += 1
        fig.show()
        plt.close('all')
    frames = []
    for i in range(bridge.shape[0]):
        frame_image = imageio.imread(save_path / 'temp' / f'{i:03d}.png')
        frames.append(frame_image)
    imageio.mimsave(save_path / save_name, frames, duration=0.2)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)