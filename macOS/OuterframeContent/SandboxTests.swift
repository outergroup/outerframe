// Sandbox tests: uncomment code to test
//
// Many of these tests cause the process to crash when the sandbox blocks them.
// The tests are commented out to prevent crashes during normal operation.
// Uncomment specific tests as needed for testing sandbox restrictions.

//import UserNotifications
//
//func testSandboxRestrictions() {
//    print("\n=== Testing UI Creation in Sandbox ===")
//
//    // Test 1: Try to create an NSWindow
////    print("Test 1: Attempting to create NSWindow...")
////    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 200),
////                         styleMask: [.titled, .closable],
////                         backing: .buffered,
////                         defer: false)
////    window.title = "Sandbox Test Window"
////    window.makeKeyAndOrderFront(nil)
////    print("❌ SECURITY ISSUE: NSWindow creation succeeded! Window: \(window)")
////
////    // Test 2: Try to access NSApplication.shared
////    print("\nTest 2: Attempting to access NSApplication.shared...")
////    let app = NSApplication.shared
////    print("❌ SECURITY ISSUE: NSApplication.shared access succeeded! App: \(app)")
////
////    // Try to show an alert
////    print("\nTest 3: Attempting to show an NSAlert...")
////    let alert = NSAlert()
////    alert.messageText = "Security Test"
////    alert.informativeText = "This alert should not appear!"
////    alert.addButton(withTitle: "OK")
////    let response = alert.runModal()
////    print("❌ SECURITY ISSUE: NSAlert shown! Response: \(response)")
//
//    // Test 4: Try to access the pasteboard
//    print("\nTest 4: Attempting to access NSPasteboard...")
//    let pasteboard = NSPasteboard.general
//    let originalChangeCount = pasteboard.changeCount
//
//    // Try to clear and write to pasteboard
//    let clearResult = pasteboard.clearContents()
//    let writeResult = pasteboard.setString("Security test", forType: .string)
//
//    // Check if the operations actually succeeded
//    let newChangeCount = pasteboard.changeCount
//    let contents = pasteboard.string(forType: .string)
//
//    if clearResult == 0 || !writeResult || newChangeCount == originalChangeCount {
//        print("✅ NSPasteboard access blocked (operations failed)")
//    } else {
//        print("❌ SECURITY ISSUE: NSPasteboard access succeeded! Contents: \(contents ?? "nil")")
//    }
//
//    // Test 5: Try to access screen
////    print("\nTest 5: Attempting to access NSScreen...")
////    if let screen = NSScreen.main {
////        print("❌ SECURITY ISSUE: NSScreen.main access succeeded! Screen: \(screen)")
////        print("  Screen size: \(screen.frame.size)")
////        print("  Screen scale: \(screen.backingScaleFactor)")
////    } else {
////        print("✅ NSScreen.main access blocked")
////    }
//
//    // Test 6: Try to create a status bar item
////    print("\nTest 6: Attempting to create NSStatusBar item...")
////    let statusBar = NSStatusBar.system
////    let statusItem = statusBar.statusItem(withLength: NSStatusItem.variableLength)
////    statusItem.button?.title = "Test"
////    print("❌ SECURITY ISSUE: NSStatusBar item created! Item: \(statusItem)")
//
//    // Test 7: Try to open a URL
//    print("\nTest 7: Attempting to open URL...")
//    let testURL = URL(string: "https://example.com")!
//    if NSWorkspace.shared.open(testURL) {
//        print("❌ SECURITY ISSUE: URL opened successfully!")
//    } else {
//        print("✅ URL opening blocked")
//    }
//
//
//    // Test 8: Try to access file dialogs
////    print("\nTest 8: Attempting to create NSOpenPanel...")
////    let openPanel = NSOpenPanel()
////    openPanel.message = "This should not appear"
////    print("❌ SECURITY ISSUE: NSOpenPanel created! Panel: \(openPanel)")
//
//    // Test 9: Try to post distributed notifications (cross-process)
//    print("\nTest 9: Attempting to post distributed notification...")
//    let distributedCenter = DistributedNotificationCenter.default()
//    distributedCenter.post(name: NSNotification.Name("TestDistributedNotification"), object: nil)
//    print("⚠️  NOTE: post() returns void - cannot detect if sandbox blocked delivery")
//    print("   Sandbox should block: com.apple.distributed_notifications@*")
//
//    // Test 10: Try to access workspace notifications
//    print("\nTest 10: Attempting to access workspace notifications...")
//    let workspaceCenter = NSWorkspace.shared.notificationCenter
//    workspaceCenter.post(name: NSWorkspace.didLaunchApplicationNotification, object: nil)
//    print("⚠️  NOTE: post() returns void - cannot detect if sandbox blocked delivery")
//    print("   Sandbox should block workspace notification services")
//
//    // Test 11: Try to show user notifications
//    print("\nTest 11: Attempting to show user notification...")
//    if #available(macOS 10.14, *) {
//        let content = UNMutableNotificationContent()
//        content.title = "Security Test"
//        content.body = "This notification should not appear"
//        let request = UNNotificationRequest(identifier: "test", content: content, trigger: nil)
//
//        UNUserNotificationCenter.current().add(request) { error in
//            if let error = error {
//                print("✅ User notification blocked: \(error)")
//            } else {
//                print("❌ SECURITY ISSUE: User notification request succeeded!")
//            }
//        }
//    }
//
//    print("\n=== UI Creation Tests Complete ===\n")
//}
//
// DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
//     testSandboxRestrictions()
// }
