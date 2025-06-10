# outerframe

An [outerframe](https://probablymarcus.com/blocks/2025/06/08/the-web-could-use-machine-code.html) is like a web view that can run machine code.


## Performance demo

A working performance demo is in the [MacPerfDemo](MacPerfDemo) folder. It shows how a proper outerframe would perform.

Feel free to run it yourself, but you shouldn't run it if you don't trust me (Marcus), or without first vetting the code. The app currently runs unsandboxed, for boring reasons. (I still need to figure out how to send mach ports between processes in a sandbox.)

To keep this focused, I don't actually let this perf demo run external code. The app does render visualizations in a background process, but the visualization code is packaged into this project.

To see the demos, build the app in Xcode, then navigate the app to:
- https://probablymarcus.com/stuff/intro.outerdoc
- https://probablymarcus.com/stuff/distributions.outerdoc
- https://probablymarcus.com/stuff/distributions-1d.outerdoc
- https://probablymarcus.com/stuff/numbers.outerdoc
- https://probablymarcus.com/stuff/checkerboard.outerdoc
- https://probablymarcus.com/stuff/sine.outerdoc
