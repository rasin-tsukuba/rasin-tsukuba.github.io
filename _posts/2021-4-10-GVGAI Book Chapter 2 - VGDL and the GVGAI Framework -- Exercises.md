---
layout: post
title: GVGAI Book Chapter 2 - VGDL and the GVGAI Framework -- Exercises
subtitle: 
date: 2021-04-10
author: Rasin
header-img: img/aliens3.png
catalog: true
tags:
  - Games
  - GVGAI
  - VGDL
---

[书籍网站](https://gaigresearch.github.io/gvgaibook/)

[本章原文](https://gaigresearch.github.io/gvgaibook/PDF/chapters/ch02.pdf?raw=true)

[本章练习](https://gaigresearch.github.io/gvgaibook/PDF/exercises/exercises02.pdf?raw=true)

------

# GVGAI环境配置

Github中提供了[GVGAI框架](https://github.com/GAIGResearch/GVGAI)。这里使用与书中匹配的[2.3](https://github.com/GAIGResearch/GVGAI/releases/tag/2.3)版本来进行练习。

## 下载并安装 IntelliJ

IntelliJ[下载地址](https://www.jetbrains.com/idea/download/#section=windows)与 JDK [下载地址](https://www.oracle.com/java/technologies/javase-downloads.html)。

下载安装完成后导入项目，如图所示：

![项目总览](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20210410195844.png)

## 运行项目

首先我们先试着用GVGAI运行游戏。可以使用键盘直接控制游玩，或者使用示例agent来自动测试。

### 以玩家身份玩游戏

在左边的 `Project` 导航中选择 `src -> tracks -> singlePlayer -> Test.java` 双击打开。

为了顺利编译文件，根据一下图示顺序设定好配置。配置名可以任意选择。

![第一步](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20210410200412.png)

![第二步](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20210410200545.png)

确认之后点击右上角的绿色运行按钮或者使用快捷键 `Shift+F10` 即可开始运行。

![SinglePlayer运行画面](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20210410200814.png)

默认启动的游戏是 `Alians`，玩家通过键盘方向键`←→`和空格键进行操作。

### 更改游戏

在代码中的`29`行我们可以看到游戏集文件 `examples/all_games_sp.csv`。使用CSV插件可以看到游戏列表：

![all_games_sp](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20210410201344.png)

那么接下来在文件的 `37`行中，我们可以通过修改 `gameIdx` 来更改游戏。例如，将其修改为:

```
int gameIdx = 9;
```

再次编译运行，游戏就变成了炸弹人：

![bomberman](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20210410201650.png)


### 用机器人玩游戏

要使用示例Agent来运行集合中的游戏时，请注释掉第 `49`行（这行代码是给玩家试玩使用的），并解开第 `52` 行注释。 


```
		// 1. This starts a game, in a level, played by a human.
		//ArcadeMachine.playOneGame(game, level1, recordActionsFile, seed);

		// 2. This plays a game in a level by the controller.
		ArcadeMachine.runOneGame(game, level1, visuals, sampleRHEAController, recordActionsFile, seed, 0);
```

`runOneGame` 第四个函数指定Agent的位置（默认是 `Rolling Horizon Evolutionary Algorithm agent`）。

文件中第 `18` 至 `26`行定义了本框架中包含的实例Agents，例如 `Monte Carlo Tree Search agent` 和 `OLETS`。

# VGDL

`VGDL` 游戏都存储在 `example`文件夹中。每个游戏类别都有一个文件夹：`gridphysics` 和 `contphysics` 包含具有传统和现实世界物理引擎的单人游戏，`2player` 中都是双人游戏，`gameDesign` 包含参数化的游戏。

## 修改游戏

可以尝试在 `example/gridphysics` 中打开一个 `VGDL`文件，一般以 `.txt` 后缀结尾。学习`VGDL`不同的部分可以从修改内容的值开始。也可以根据你的想法在 `InteractionSet` 中添加新的规则或者新的贴图。有可能你也能创造新的游戏。

## 更改关卡

可以通过修改例如 `aliens_lvl0.txt` 之类的文件来更改关卡，比如关卡的布局，贴图的初始位置等等。可以给关卡增加不同难度或者创造新的关卡。

# 提交到GVGAI比赛服务器

一下是提交步骤：

* 在 [GAGVI官网](http://www.gvgai.net/) 创建账号。
* 登入并点击 `Submit → Submit to Signle Player Planning`。
* 您的控制器文件应命名为 `Agent.java`，并包含一个与您用户名相同的包，以 `.zip` 文件存储。网站中有更详尽的[提交规范](http://www.gvgai.net/submit_gvg.php)
* 填写需要的信息，并选择一个游戏集提交。您的代码将会在服务器上编译执行，具体提交信息可以在个人主页上查看。
* 一旦编译运行完成，你可以在排行版上看到您的结果。
* 如果您想参赛，可以先用示例代码提交测试。


