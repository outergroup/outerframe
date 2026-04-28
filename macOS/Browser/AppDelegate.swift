import Cocoa
import Outerframe

@main
final class AppDelegate: NSObject, NSApplicationDelegate {
    @IBOutlet private var window: NSWindow?

    private var browserWindowControllers: [OpenBrowserWindowController] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        OuterframeConfiguration.processesXPCServiceName = "dev.outergroup.OuterframeProcesses"
        OuterframeConfiguration.networkProxyXPCServiceName = "dev.outergroup.OuterframeNetworkProxy"
        OuterframeConfiguration.warmupXPCConnections()

        let controller: OpenBrowserWindowController
        if let window {
            controller = OpenBrowserWindowController(window: window)
        } else {
            controller = OpenBrowserWindowController()
        }
        registerWindowController(controller)
        controller.showWindow(nil)
        setupFileMenuItems()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
        true
    }

    func registerWindowController(_ controller: OpenBrowserWindowController) {
        browserWindowControllers.removeAll { $0 === controller }
        browserWindowControllers.append(controller)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(windowWillClose(_:)),
                                               name: NSWindow.willCloseNotification,
                                               object: controller.window)
    }

    @IBAction func newDocument(_ sender: Any?) {
        activeWindowController()?.createNativeTab(focusAddressBar: true)
    }

    @IBAction func openDocument(_ sender: Any?) {
        activeWindowController()?.focusAddressBar(sender)
    }

    @IBAction func openLocation(_ sender: Any?) {
        activeWindowController()?.focusAddressBar(sender)
    }

    private func setupFileMenuItems() {
        guard let fileMenu = NSApp.mainMenu?.item(withTitle: "File")?.submenu else { return }

        let newTabItem = setupNewTabMenuItem(in: fileMenu)
        clearDuplicateNewTabKeyEquivalents(except: newTabItem, in: NSApp.mainMenu)
        setupOpenLocationMenuItem(in: fileMenu)
    }

    private func setupNewTabMenuItem(in fileMenu: NSMenu) -> NSMenuItem? {
        guard let item = fileMenu.items.first(where: { $0.action == #selector(newDocument(_:)) }) else { return nil }

        item.title = "New Tab"
        item.target = self
        item.keyEquivalent = "t"
        item.keyEquivalentModifierMask = .command
        return item
    }

    private func clearDuplicateNewTabKeyEquivalents(except newTabItem: NSMenuItem?, in menu: NSMenu?) {
        guard let menu else { return }

        for item in menu.items {
            if item !== newTabItem,
               item.keyEquivalent.lowercased() == "t",
               item.keyEquivalentModifierMask.intersection([.command, .option, .control, .shift]) == .command {
                item.keyEquivalent = ""
            }

            clearDuplicateNewTabKeyEquivalents(except: newTabItem, in: item.submenu)
        }
    }

    private func setupOpenLocationMenuItem(in fileMenu: NSMenu) {
        guard !fileMenu.items.contains(where: { $0.action == #selector(openLocation(_:)) }) else { return }

        let item = NSMenuItem(title: "Open Location...",
                              action: #selector(openLocation(_:)),
                              keyEquivalent: "l")
        item.target = self
        item.keyEquivalentModifierMask = .command

        let insertIndex = fileMenu.items.firstIndex { $0.action == #selector(openDocument(_:)) }
            .map { $0 + 1 }
            ?? min(1, fileMenu.items.count)
        fileMenu.insertItem(item, at: insertIndex)
    }

    private func activeWindowController() -> OpenBrowserWindowController? {
        if let controller = NSApp.keyWindow?.windowController as? OpenBrowserWindowController {
            return controller
        }
        return browserWindowControllers.last
    }

    @objc private func windowWillClose(_ notification: Notification) {
        guard let window = notification.object as? NSWindow else { return }
        NotificationCenter.default.removeObserver(self,
                                                  name: NSWindow.willCloseNotification,
                                                  object: window)
        browserWindowControllers.removeAll { $0.window === window }
    }
}
