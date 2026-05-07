# outerframe

An outerframe is like a web view that can run machine code.

This repo contains a working outerframe implementation for macOS. [Outer Loop](https://outerloop.sh/) uses this code.

Read more:
- June 2025: ["The web could use machine code"](https://probablymarcus.com/blocks/2025/06/08/the-web-could-use-machine-code.html)
- April 2026: ["It's like a web view, but native"](https://probablymarcus.com/blocks/2026/04/28/like-a-web-view-but-native.html)

There are two ways you might play with this:
- embed the outerframe in your own native macOS apps
- build apps/content to go inside of an outerframe


## How to create outerframe content

The easiest way is to start with an existing project and alter it.

Example outerframe content:
- Run [Top](https://github.com/outergroup/top) backend
- Clone example projects in [Swift](https://github.com/outergroup/hello-outerframe-macOS-swift) or [C](https://github.com/outergroup/hello-outerframe-macOS-objc) and vibe-code something for yourself.
- Browse the Outerframe Cookbook:
  - Swift: https://probablymarcus.com/cookbook-swift.outer  (view in Outer Loop or Outer Frame)
  - C: https://probablymarcus.com/cookbook-objc.outer  (view in Outer Loop or Outer Frame)
  - Source: [outerframe-cookbook](https://github.com/outergroup/outerframe-cookbook)


## How to play with outerframe content (existing or yours)

You can open it in Outer Loop, or you can build your own minimal "Outer Frame" browser in this repo.

To build your own browser, in the [macOS](macOS) folder, open Outerframe.xcodeproj (which will launch Xcode) and build the target "Outer Frame". You can now navigate this to arbitrary URLs hosting outerframe content.


## How to use an outerframe in your own project

To use this, add [macOS/Outerframe.xcodeproj](macOS/Outerframe.xcodeproj) to your own Xcode project. Use [macOS/Browser](macOS/Browser) as example code. The key components for hosts are the `OuterframeView` and the network proxy.
