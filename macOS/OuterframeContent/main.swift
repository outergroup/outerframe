import Darwin
import Foundation
import LibOuterframeContent

OuterframeContentRuntime.configureOutputBuffering()

struct SpawnedOuterframeContentLaunch {
    let infrastructureSocketFD: Int32
    let pluginSocketFD: Int32
    let networkProxyPort: UInt16?
    let hostBundleIdentifier: String
}

func parseSpawnedLaunch() -> SpawnedOuterframeContentLaunch {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.contains("--infra-socket-fd") else {
        print("OuterframeContent: Usage: OuterframeContent --infra-socket-fd <fd> --plugin-socket-fd <fd> --host-bundle-id <bundle-id> [--network-proxy-port <port>]")
        exit(1)
    }

    var infrastructureSocketFD: Int32?
    var pluginSocketFD: Int32?
    var networkProxyPort: UInt16?
    var hostBundleIdentifier: String?

    var iterator = args.makeIterator()
    while let argument = iterator.next() {
        switch argument {
        case "--infra-socket-fd":
            guard let value = iterator.next(), let parsed = Int32(value) else {
                print("OuterframeContent: --infra-socket-fd requires an integer value")
                exit(1)
            }
            infrastructureSocketFD = parsed

        case "--plugin-socket-fd":
            guard let value = iterator.next(), let parsed = Int32(value) else {
                print("OuterframeContent: --plugin-socket-fd requires an integer value")
                exit(1)
            }
            pluginSocketFD = parsed

        case "--network-proxy-port":
            guard let value = iterator.next(), let parsed = UInt16(value) else {
                print("OuterframeContent: --network-proxy-port requires an integer value")
                exit(1)
            }
            networkProxyPort = parsed

        case "--host-bundle-id":
            guard let value = iterator.next(), !value.isEmpty else {
                print("OuterframeContent: --host-bundle-id requires a non-empty value")
                exit(1)
            }
            hostBundleIdentifier = value

        default:
            continue
        }
    }

    guard let infrastructureSocketFD else {
        print("OuterframeContent: Must include --infra-socket-fd")
        exit(1)
    }
    guard let pluginSocketFD else {
        print("OuterframeContent: Must include --plugin-socket-fd")
        exit(1)
    }
    guard let hostBundleIdentifier else {
        print("OuterframeContent: Must include --host-bundle-id")
        exit(1)
    }

    return SpawnedOuterframeContentLaunch(infrastructureSocketFD: infrastructureSocketFD,
                                          pluginSocketFD: pluginSocketFD,
                                          networkProxyPort: networkProxyPort,
                                          hostBundleIdentifier: hostBundleIdentifier)
}

let launch = parseSpawnedLaunch()
OuterframeContentRuntime.run(infrastructureSocketFD: launch.infrastructureSocketFD,
                             pluginSocketFD: launch.pluginSocketFD,
                             networkProxyPort: launch.networkProxyPort,
                             hostBundleIdentifier: launch.hostBundleIdentifier)
