import AppKit

@MainActor
public protocol OuterframeViewDelegate: AnyObject {
    func outerframeView(_ view: OuterframeView,
                        didRequestOpenWindowWithURLString urlString: String,
                        displayString: String?,
                        preferredSize: CGSize?)
    func outerframeView(_ view: OuterframeView, didRequestNavigateToURLString urlString: String)
    func outerframeView(_ view: OuterframeView,
                        didRequestOpenTabWithURLString urlString: String,
                        displayString: String?)
    func outerframeView(_ view: OuterframeView, didRequestHistoryPushEntryWithID entryID: UUID, urlString: String?)
    func outerframeView(_ view: OuterframeView, didRequestHistoryReplaceEntryWithID entryID: UUID, urlString: String?)
    func outerframeView(_ view: OuterframeView, didRequestHistoryGo delta: Int32)
    func outerframeView(_ view: OuterframeView, didSetTitle title: String?)
    func outerframeView(_ view: OuterframeView, didSetIcon icon: NSImage?)
    func outerframeViewDidResetOuterframeContentOutput(_ view: OuterframeView)
    func outerframeView(_ view: OuterframeView, didReceiveOuterframeContentStdout text: String)
    func outerframeView(_ view: OuterframeView, didReceiveOuterframeContentStderr text: String)
    func outerframeViewDidTimeoutWaitingForPluginLoaded(_ view: OuterframeView)
}

public extension OuterframeViewDelegate {
    func outerframeView(_ view: OuterframeView, didRequestNavigateToURLString urlString: String) {}
    func outerframeView(_ view: OuterframeView,
                        didRequestOpenTabWithURLString urlString: String,
                        displayString: String?) {}
    func outerframeView(_ view: OuterframeView, didRequestHistoryPushEntryWithID entryID: UUID, urlString: String?) {}
    func outerframeView(_ view: OuterframeView, didRequestHistoryReplaceEntryWithID entryID: UUID, urlString: String?) {}
    func outerframeView(_ view: OuterframeView, didRequestHistoryGo delta: Int32) {}
    func outerframeView(_ view: OuterframeView, didSetTitle title: String?) {}
    func outerframeView(_ view: OuterframeView, didSetIcon icon: NSImage?) {}
}
