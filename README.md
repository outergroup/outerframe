# outerframe

An outerframe is like a web view that can run machine code.

This repo contains a working outerframe implementation for macOS. [Outer Loop](https://outerloop.sh/) uses this code.

Read more:
- June 2025: ["The web could use machine code"](https://probablymarcus.com/blocks/2025/06/08/the-web-could-use-machine-code.html)
- April 2026: ["It's like a web view, but native"](https://probablymarcus.com/blocks/2026/04/28/like-a-web-view-but-native.html)


## How to play with your own built outerframe

Open [macOS/Outerframe.xcodeproj](macOS/Outerframe.xcodeproj) and build the target "Outer Frame". This is a simple browser built on the outerframe. You can now navigate this to arbitrary URLs hosting outerframe content.

Example outerframe content:
- Run [Top](https://github.com/outergroup/top) backend
  - If you run it on a remote machine, use port forwarding, or use [Outer Loop](https://outerloop.sh)
- Clone [hello-macOS-outerframe](https://github.com/outergroup/hello-macOS-outerframe) and vibe-code something for yourself.
- Play with an early [outerframe-cookbook](https://github.com/outergroup/outerframe-cookbook)


## How to use an outerframe in your own project

To use this, add [macOS/Outerframe.xcodeproj](macOS/Outerframe.xcodeproj) to your own Xcode project. Use [macOS/Browser](macOS/Browser) as example code. The key components for hosts are the `OuterframeView` and the network proxy.
