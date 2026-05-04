import AppKit

@MainActor
public protocol OuterframeViewDelegate: AnyObject {
    func outerframeView(_ view: OuterframeView,
                        didRequestOpenWindowWithURLString urlString: String,
                        displayString: String?,
                        preferredSize: CGSize?)
    func outerframeViewDidResetOuterframeContentOutput(_ view: OuterframeView)
    func outerframeView(_ view: OuterframeView, didReceiveOuterframeContentStdout text: String)
    func outerframeView(_ view: OuterframeView, didReceiveOuterframeContentStderr text: String)
    func outerframeViewDidTimeoutWaitingForPluginLoaded(_ view: OuterframeView)
}
