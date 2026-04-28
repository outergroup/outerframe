import Foundation

public enum OuterframeDebugSettings {
    private static let debugModeEnabledKey = "OuterframeDebugModeEnabled"
    private static let autoResumeKey = "OuterframeDebugAutoResume"

    public static let debugModeDidChangeNotification = Notification.Name("OuterframeDebugModeDidChange")
    public static let autoResumeDidChangeNotification = Notification.Name("OuterframeDebugAutoResumeDidChange")

    public static var isDebugModeEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: debugModeEnabledKey) }
        set {
            let defaults = UserDefaults.standard
            let previous = defaults.bool(forKey: debugModeEnabledKey)
            guard previous != newValue else { return }
            defaults.set(newValue, forKey: debugModeEnabledKey)
            NotificationCenter.default.post(name: debugModeDidChangeNotification, object: nil)
        }
    }

    public static var shouldAutoResumeOnDebuggerAttach: Bool {
        get {
            if let storedValue = UserDefaults.standard.object(forKey: autoResumeKey) as? Bool {
                return storedValue
            }
            return true
        }
        set {
            let defaults = UserDefaults.standard
            let previous = defaults.object(forKey: autoResumeKey) as? Bool ?? true
            guard previous != newValue else { return }
            defaults.set(newValue, forKey: autoResumeKey)
            NotificationCenter.default.post(name: autoResumeDidChangeNotification, object: nil)
        }
    }
}
