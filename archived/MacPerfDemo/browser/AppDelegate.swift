//
//  AppDelegate.swift
//

import Cocoa
import OuterdocViewer

@main
class AppDelegate: NSObject, NSApplicationDelegate {

    @IBOutlet var window: NSWindow!
    var browserContentView: BrowserContentView!

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        _ = OuterContentConnection.shared
        createNewBrowserWindow()
    }
    
    /// Creates and displays a new browser window
    func createNewBrowserWindow() {
        // Create the browser view controller
        browserContentView = BrowserContentView()
        
        // Create the window controller and configure window
        let viewController = NSViewController()
        viewController.view = browserContentView
        
        let window = NSWindow(contentViewController: viewController)
        
        // Create toolbar
        let toolbar = NSToolbar(identifier: "BrowserToolbar")
        toolbar.delegate = browserContentView
        toolbar.displayMode = .iconOnly
        window.toolbar = toolbar
        // Set window size and position
        window.setContentSize(NSSize(width: 1200, height: 700))
        window.center()
        
        // Set the browser content view as the window delegate to handle cleanup
        window.delegate = browserContentView
        
        // No title for browser window
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.toolbarStyle = .unified

        // Safari-like appearance
        window.backgroundColor = NSColor.textBackgroundColor
        
        // Show window
        window.makeKeyAndOrderFront(nil)
        
        self.window = window
    }

    func applicationWillTerminate(_ aNotification: Notification) {
        OuterContentConnection.shared.cleanup()
    }

    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
        return true
    }
    
    // Create a new window when the app is activated and no windows are open
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            createNewBrowserWindow()
        }
        return true
    }
    
    // Handle "New Window" from the File menu
    @IBAction func newDocument(_ sender: Any?) {
        createNewBrowserWindow()
    }
}
