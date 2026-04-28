import AppKit

public struct OuterframeStartPageMetadata {
    public let title: String?
    public let icon: NSImage?
    public let iconSize: CGSize?

    public init(title: String?, icon: NSImage?, iconSize: CGSize?) {
        self.title = title
        self.icon = icon
        self.iconSize = iconSize
    }
}

@MainActor
public protocol OuterframeViewDelegate: AnyObject {
    func outerframeView(_ view: OuterframeView, didUpdatePageTitle title: String?)
    func outerframeView(_ view: OuterframeView, didUpdateFavicon icon: NSImage?)
    func outerframeView(_ view: OuterframeView, didUpdateStartPageMetadata metadata: OuterframeStartPageMetadata)
    func outerframeView(_ view: OuterframeView,
                        didRequestOpenWindowWithURLString urlString: String,
                        displayString: String?,
                        preferredSize: CGSize?)
    func outerframeViewDidResetOuterframeContentOutput(_ view: OuterframeView)
    func outerframeView(_ view: OuterframeView, didReceiveOuterframeContentStdout text: String)
    func outerframeView(_ view: OuterframeView, didReceiveOuterframeContentStderr text: String)
    func outerframeViewDidTimeoutWaitingForPluginLoaded(_ view: OuterframeView)
}
