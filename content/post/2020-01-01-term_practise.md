---
title: "终端实战"
date: 2020-01-01T22:00:47+08:00
lastmod: 2020-01-02T17:54:04+08:00
draft: false
tags: ["iterm2","zsh","oh-my-zsh"]
categories: ["实用工具"]
author: "禹过留声"

---

本文主要介绍终端和 shell 的一些基本概念，并着重介绍了 zsh - 目前最强大方便的 shell 以及它的配置，希望可以提高我们码农的工作效率。

<!--more-->



## 终端
终端 `Terminal` 是一种用于与计算机进行交互的输入输出设备，它本身不提供运算处理功能。在大型机和小型机的时代，计算机是非常昂贵巨大的。通常计算机会被安置在单独的房间里，而操作计算机的人在另外的房间通过终端设备 `TTY` / `CONSOLE` 与计算机进行交互。现在由于个人电脑的普及，基本很难看到专门的终端设备。负责输入的键盘，负责输出的显示器，再加上一个与硬件基本无关的虚拟终端`终端模拟器`这三者共同构成了传统意义上的终端。

> **TIPS: 埃尼阿克**
> 
>  埃尼阿克 `ENIAC` 是世界上第一台现代电子数字计算机，诞生于1946 年 2 月 14 号美国宾夕法尼亚大学。
>  它长 30.48 米，宽 6 米，高 2.4 米，占地面积约 170 平方米，30 个操作台，重达 30 英吨，耗电量 150 千瓦，造价 48 万美元。计算速度是每秒 5000 次加法或 400 次乘法。
### 终端 `TTY`
`TTY` 是 TeleTYpe 的缩写，叫电传打字机，一个类似电报机的设备。这个也就是最早期的终端。
![Alt text](/imgs/term_practise-tty.jpg)
它原本的用途是在电报线路上收发电报，但鉴于它既能通过键盘发送信号，又能将接受到的信号打印在纸带上，最最最重要的是价格低廉，它就被 Unix 的创始人 Ken Thompson 和 Dennis Ritchie 用于连接到计算机上，让多个用户都可以通过终端登陆操作主机，所以它就成了第一个 Unix 终端。

### 控制台 `CONSOLE`
`CONSOLE` 是控制台的意思，它是一种特殊的终端，特殊的地方是它和计算机主体是一体的，是专门用来管理主机的，只能给系统管理员使用，有着比普通终端高的权限。一般一台计算机上只有一个控制台，但可以连很多终端。 `CONSOLE` 和 `TTY` 都算是终端，硬要说区别就是亲儿子和干儿子或 root 和非 root 用户的关系。

###  终端模拟器
终端模拟器 `Terminal Emulator` 也叫终端仿真器。它加上键盘和显示器共同构建了以前的终端。它的工作流程如下：

1.  捕获键盘输入（ STDIN ）
2.  将输入发送给命令行程序（ SHELL ）
3.  拿到命令行程序的输出结果（ STDOUT 和 STDERR ）
4.  调用图形接口，将输出结果渲染到显示器上
#### 终端窗口和虚拟控制台
终端模拟器分为两种，一种是终端窗口，就是我们一般运行在图形用户界面里的，像 GNU/Linux 下的 `gnome-termial`， mac 下的 `iterm2`, windows 下的 `wsl-terminal`。另一种叫虚拟控制台，像 Ubuntu 系统中，通过 `Ctrl`+`Alt`+`F1,F2...F6` 等组合键可以切换出全屏的终端界面（ `Ctrl`+`Alt`+`F7` 可以切回图形界面），这就是虚拟控制台。它是直接由操作系统内核直接提供的。

###  实用终端
####  mac 神器 - iterm2
*  安装
```bash
# 通过brew安装
brew cask install iterm2
# 通过iterm2官网下载安装
```
* 配置
	* 字体 `Courier New` + `meslo`
  ```bash
  # 拉取字体厂库
  git clone https://github.com/powerline/fonts.git --depth=1
  # install 安裝
  cd fonts  &&  ./install.sh
  ```

	* 终端配色
  ```bash
  # 拉取终端配色厂库
  git clone https://github.com/mbadolato/iTerm2-Color-Schemes.git --depth=1
  ```

	![Alt text](/imgs/term_practise-color_schema.png)

	* 终端显示行数
  
  ![Alt text](/imgs/term_practise-scrollback_lines.png)

## shell
`shell` 也叫命令解释器，它通过解析命令来调用系统调用和 API 来操作内核，进而读写硬件完成任务。

![Alt text](/imgs/term_practise-shell.png)

它可以通过图形化 shell，像 windows 里的文件管理器 `Explorer.exe`, Linux 的桌面环境 `GNOME`， `KDE` 等完成操作，也可以通过命令行 shell，像 windows 里的 `cmd.exe`, Linux 里的 `bash`, `zsh` 等。

### zsh
`bash` 在 2019 年前几乎是所有类型操作系统的默认 shell，但是 2019 年 Mac 的 Catalina 将 `zsh` 设置为默认的 shell。为什么？因为 `zsh` 完全兼容 `bash`，并且提供更多的功能。它提供以下功能：

1. 内置的拼写校正
2. 改进的命令行完成
3. 主题支持
4. 各种各样的可加载插件
其安装和切换命令如下：
```bash
# 安装
brew install zsh
# 切换
chsh -s /bin/zsh
```

### oh-my-zsh
`oh-my-zsh` 是一个工具，它可以帮助用户更轻松的启用 zsh 插件，在预制的主题间切换，快速自定义 shell。
* 安装 `oh-my-zsh`
```bash
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```
* 配置主题 annoster
#### 插件
插件为 zsh 提供了无限可能。
* `git`

默认安装，会提示分支等信息

* 语义高亮 `zsh-syntax-highlighting`

```bash
# 
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

*  自动提示 `zsh-autosuggestions`

```bash
git clone git://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
```

* 自动跳转 `autojump`

```bash
# 安装autojump
brew install autojump
# 在~/.zshrc中添加
[ -f /usr/local/etc/profile.d/autojump.sh ] && . /usr/local/etc/profile.d/autojump.sh
```
![Alt text](/imgs/term_practise-zsh_terminal.png)






## 参考文献
1. [命令行界面 (CLI)、终端 (Terminal)、Shell、TTY，傻傻分不清楚？][blog1]
2. [iterm2官网][iterm2]
3. [iTerm2 + Oh My Zsh 打造舒适终端体验][oh-my-zsh]
4. [Mac终端工具][mac_tower]


[blog1]: https://printempw.github.io/the-difference-between-cli-terminal-shell-tty/#2-2-%E6%8E%A7%E5%88%B6%E5%8F%B0-Console-%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F
[iterm2]:https://www.iterm2.com/downloads.html
[oh-my-zsh]:https://segmentfault.com/a/1190000014992947#item-1
[mac_tower]:https://p-dw2s.tower.im/p/87dc

