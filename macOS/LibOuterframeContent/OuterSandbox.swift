import Foundation

// Declare the sandbox C functions we need
@_silgen_name("sandbox_init_with_parameters")
func sandbox_init_with_parameters(_ profile: UnsafePointer<CChar>, _ flags: UInt64, _ parameters: UnsafePointer<UnsafePointer<CChar>?>, _ errorbuf: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32

@_silgen_name("sandbox_check")
func sandbox_check(_ pid: pid_t, _ operation: UnsafePointer<CChar>?, _ type: Int32, _ args: CVarArg...) -> Int32

public struct OuterSandbox {
    private static func makeProfile(networkRule: String, allowedReadPaths: [String]) -> String {
        let readPathRules = allowedReadPaths.isEmpty ? "" : """

        ;; Additional allowed read paths (e.g., local bundle for debugging)
        (allow file-read* file-test-existence file-map-executable
            \(allowedReadPaths.map { "(subpath \"\($0)\")" }.joined(separator: "\n        ")))
        """

        return """
        (version 1)
        
        ;; Parameters: HOME_DIR, APP_BUNDLE_ID
        
        ;; Base sandbox policy - deny everything by default
        (deny default)
        (import "/System/Library/Sandbox/Profiles/bsd.sb")
        
        ;; Process introspection
        (allow process-info* (target self))
        (allow sysctl-read)
        
        ;; ===== FILESYSTEM ACCESS =====
        
        ;; System libraries and frameworks
        (allow file-read* file-test-existence
            (subpath "/Library/Apple")
            (subpath "/Library/Filesystems/NetFSPlugins")
            (subpath "/Library/Preferences/Logging")
            (subpath "/System")
            (literal "/private/var/db/DarwinDirectory/local/recordStore.data")
            (subpath "/private/var/db/timezone")
            (subpath "/usr/lib")
            (subpath "/usr/share"))
        
        ;; Allow mapping system frameworks and libraries
        (allow file-map-executable
            (subpath "/Library/Apple/System/Library/Frameworks")
            (subpath "/Library/Apple/System/Library/PrivateFrameworks")
            (subpath "/Library/Apple/usr/lib")
            (subpath "/System/Library/Frameworks")
            (subpath "/System/Library/PrivateFrameworks")
            (subpath "/usr/lib"))
        
        ;; Basic filesystem navigation
        (allow file-read-metadata file-test-existence
            (literal "/")
            (literal "/etc")
            (literal "/tmp")
            (literal "/var")
            (literal "/private/etc/localtime"))
        
        ;; System configuration files
        (allow file-read* file-test-existence
            (literal "/dev/autofs_nowait")
            (literal "/dev/random")
            (literal "/dev/urandom")
            (literal "/private/etc/master.passwd")
            (literal "/private/etc/passwd")
            (literal "/private/etc/protocols")
            (literal "/private/etc/services")
            (literal "/Library/Preferences/com.apple.networkd.plist")
            (literal "/Library/Preferences/com.apple.security.plist"))
        
        ;; Device files
        (allow file-read* file-test-existence file-write-data
            (literal "/dev/null")
            (literal "/dev/zero"))
        
        ;; Temporary directories (for downloads and caches)
        (allow file-read* file-test-existence file-write*
            (subpath "/private/tmp")
            (subpath "/private/var/folders"))
        
        ;; Application resources
        (allow file-read* file-test-existence
            (subpath "/Applications"))
        
        ;; User preferences and caches
        (allow file-read*
            (subpath (string-append (param "HOME_DIR") "/Library/Preferences")))
        
        ;; Application cache directory
        (allow file-read* file-test-existence file-write* file-ioctl
            (subpath (string-append (param "HOME_DIR") "/Library/Caches/" (param "APP_BUNDLE_ID"))))
        
        ;; Outerframe bundle cache
        (allow file-read* file-test-existence file-map-executable
            (subpath (string-append (param "HOME_DIR") "/Library/Containers/" (param "APP_BUNDLE_ID") "/Data/Library/" (param "APP_BUNDLE_ID"))))
        
        ;; Graphics and media resources
        (allow file-read*
            (subpath "/Library/GPUBundles")
            (subpath "/Library/Video/Plug-Ins")
            (subpath "/System/Library/ColorSync/Profiles")
            (subpath "/System/Library/Components/AudioCodecs.component")
            (subpath "/System/Library/CoreServices/RawCamera.bundle")
            (subpath "/System/Library/Extensions")
            (subpath "/System/Library/Video/Plug-Ins"))
        
        ;; Code signing verification
        (allow file-read* file-write-unlink
            (prefix "/private/tmp/cvmsCodeSignObj"))
        (allow file-read*
            (extension "com.apple.cvms.kernel")
            (prefix "/private/var/db/CVMS/cvmsCodeSignObj"))
        
        ;; MessageTracer for diagnostics
        (allow file-read-data
            (path "/Library/MessageTracer/SubmitDiagInfo.default.domains.searchtree")
            (path "/System/Library/MessageTracer/SubmitDiagInfo.default.domains.searchtree"))
        
        ;; Metal shader compilation
        (allow file-issue-extension
            (require-all
                (extension-class "com.apple.app-sandbox.read")
                (subpath "/")))
        
        ;; ===== USER PREFERENCES =====
        
        (allow user-preference-read)
        
        ;; System preferences
        (allow user-preference-read
            (preference-domain "com.apple.loginwindow")
            (preference-domain "com.apple.MCX")
            (preference-domain "kCFPreferencesAnyApplication"))
        
        ;; App-specific preferences
        (allow user-preference-read user-preference-write
            (preference-domain-regex #"^dev\\.outergroup\\."))
        
        ;; ===== IPC AND XPC =====

        (allow ipc-posix-shm)
        (allow ipc-posix-shm-read*
            (ipc-posix-name-prefix "apple.cfprefs."))
        ;;(allow ipc-posix-shm-read-data
        ;;    (ipc-posix-name "apple.shm.notification_center"))
        
        ;; System services
        (allow mach-lookup
            (global-name "com.apple.audio.AudioComponentRegistrar")
            (global-name "com.apple.bsd.dirhelper")
            (global-name "com.apple.CARenderServer")
            (global-name "com.apple.cfprefsd.agent")
            (global-name "com.apple.cfprefsd.daemon")
            (global-name "com.apple.CoreServices.coreservicesd")
            (global-name "com.apple.cvmsServ")
            (global-name "com.apple.gpumemd.source")
            (global-name "com.apple.lsd.mapdb")
            (global-name "com.apple.lsd.modifydb")
            (global-name "com.apple.powerlog.plxpclogger.xpc")
            (global-name "com.apple.PowerManagement.control")
            (global-name "com.apple.SecurityServer")
            (global-name "com.apple.system.opendirectoryd.membership")
            (global-name "com.apple.tsm.uiserver")
            (global-name "com.apple.trustd.agent")
            (global-name "com.apple.appleneuralengine"))
        
        ;; Prevent direct creation of UI
        (deny mach-lookup
            (global-name "com.apple.windowserver.active")
            (global-name "com.apple.HIToolbox.xpc")
            (global-name "com.apple.coreservices.launchservicesd")
            (global-name "com.apple.system.notification_center"))
        
        ;; Prevent distributed notifications (cross-process communication)
        (deny mach-lookup
            (global-name-regex #"^com\\.apple\\.distributed_notifications@"))
        
        ;; Prevent workspace notifications
        (deny mach-lookup
            (global-name "com.apple.coreservices.sharedfilelistd")
            (global-name "com.apple.CoreServices.coreservicesd")
            (global-name "com.apple.notificationcenterui.agent"))
        
        ;; Prevent pasteboard access (clipboard read/write)
        (deny mach-lookup
            (global-name "com.apple.pasteboard.1"))
        
        ;; XPC services
        (allow mach-lookup
            (xpc-service-name "com.apple.coremedia.videodecoder")
            (xpc-service-name "com.apple.coremedia.videoencoder")
            (xpc-service-name "com.apple.MTLCompilerService")
            (xpc-service-name-regex #"\\.apple-extension-service$"))
        
        ;; ===== HARDWARE ACCESS =====
        
        ;; IOKit - Graphics and display
        (allow iokit-open
            (iokit-connection "IOAccelerator")
            (iokit-user-client-class "AGPMClient")
            (iokit-user-client-class "AppleGraphicsControlClient")
            (iokit-user-client-class "AppleGraphicsPolicyClient")
            (iokit-user-client-class "AppleIntelMEUserClient")
            (iokit-user-client-class "AppleMGPUPowerControlClient")
            (iokit-user-client-class "AppleSNBFBUserClient")
            (iokit-user-client-class "IOAccelerationUserClient")
            (iokit-user-client-class "IOFramebufferSharedUserClient")
            (iokit-user-client-class "IOHIDParamUserClient")
            (iokit-user-client-class "IOSurfaceRootUserClient")
            (iokit-user-client-class "IOSurfaceSendRight")
            (iokit-user-client-class "RootDomainUserClient")
            (iokit-user-client-class "H11ANEInDirectPathClient"))
        
        ;; Display properties
        (allow iokit-set-properties
            (require-all (iokit-connection "IODisplay")
                (require-any (iokit-property "brightness")
                    (iokit-property "linear-brightness")
                    (iokit-property "commit")
                    (iokit-property "rgcs")
                    (iokit-property "ggcs")
                    (iokit-property "bgcs"))))
        
        ;; ===== SYSTEM INFORMATION =====
        
        (allow sysctl-read
            (sysctl-name "hw.busfrequency_max")
            (sysctl-name "hw.cachelinesize")
            (sysctl-name "hw.logicalcpu_max")
            (sysctl-name "hw.memsize")
            (sysctl-name "hw.model")
            (sysctl-name "kern.osvariant_status"))
        
        ;; ===== PROCESS CONTROL =====
        
        (allow process-exec*)
        (allow process-fork)
        
        ;; ===== NETWORKING =====

        \(networkRule)
        \(readPathRules)
        """
    }
    
    public static func parameters(bundleId: String? = nil) -> [String: String] {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
        let actualBundleId = bundleId ?? Bundle.main.bundleIdentifier!

        return [
            "HOME_DIR": homeDir,
            "APP_BUNDLE_ID": actualBundleId
        ]
    }
    
    public static func isSandboxed() -> Bool {
        return sandbox_check(getpid(), nil, 0) == 1
    }
    
    public static func apply(bundleId: String? = nil,
                             allowedProxyPort: UInt16? = nil,
                             allowedReadPaths: [String] = []) -> (success: Bool, error: String?) {
        // Check if already sandboxed
        if isSandboxed() {
            return (false, "Process is already sandboxed")
        }

        // Get parameters
        let params = parameters(bundleId: bundleId)

        // Convert parameters to C string array as flat list: [key, value, key, value, ...]
        var paramStrings: [String] = []

        for (key, value) in params {
            paramStrings.append(key)
            paramStrings.append(value)
        }

        let networkRule: String
        if let allowedProxyPort {
            networkRule = """
        (allow network-outbound
            (remote tcp "localhost:\(allowedProxyPort)"))
        """
        } else {
            networkRule = """
        (deny network-outbound)
        """
        }

        let profileString = makeProfile(networkRule: networkRule, allowedReadPaths: allowedReadPaths)

        // Apply sandbox - need to keep strings alive during the call
        var errorBuf: UnsafeMutablePointer<CChar>? = nil

        // We need to ensure the parameter strings remain valid during the sandbox call
        let result = profileString.withCString { profileCString in
            // Create C string representations
            var cStringStorage: [ContiguousArray<CChar>] = []
            var cParams: [UnsafePointer<CChar>?] = []
            
            for paramString in paramStrings {
                let cString = paramString.utf8CString
                cStringStorage.append(cString)
            }
            
            // Get pointers to the C strings
            for i in 0..<cStringStorage.count {
                cStringStorage[i].withUnsafeBufferPointer { buffer in
                    cParams.append(buffer.baseAddress)
                }
            }
            cParams.append(nil) // NULL terminator
            
            return cParams.withUnsafeBufferPointer { buffer in
                sandbox_init_with_parameters(profileCString, 0, buffer.baseAddress!, &errorBuf)
            }
        }
        
        if result != 0 {
            let errorMessage = errorBuf != nil ? String(cString: errorBuf!) : "Unknown sandbox error"
            if errorBuf != nil {
                free(errorBuf)
            }
            return (false, errorMessage)
        }
        
        return (true, nil)
    }
}
