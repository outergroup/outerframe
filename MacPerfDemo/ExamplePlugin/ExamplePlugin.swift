//
//  ExamplePlugin.swift
//

import Foundation
import OuterdocPlugin


// MARK: - Plugin support
/// Plugin class to register supported file types and construct UntrustedFigure instances.
/// This class is set as the principal class for the bundle in Info.plist
///
@objc public class ExamplePlugin: NSObject, OuterdocPlugin {
    /// Text-based (outerdoc) types supported by this plugin.
    @objc public static func suppertedOuterdocContentTypes() -> [String] {
        return ["dev.outergroup.SineWaveDemo", "dev.outergroup.ScalarDistributionListDemo", "dev.outergroup.ScalarDistributionListDemoSingleCanvas", "dev.outergroup.ScalarDistributionDemo", "dev.outergroup.CheckerboardDemo", "dev.outergroup.NumbersDemo"]
    }

    @objc public static func createDynamicSubstring(contentType: String) -> OuterdocDynamicSubstring? {
        // Handle outerdoc types
        switch contentType {
        case "dev.outergroup.SineWaveDemo":
            print("Creating SineWaveDemo")
            return SineWaveSubstring(elementIds: getUniqueElementIDs())
        case "dev.outergroup.ScalarDistributionListDemo":
            print("Creating ScalarDistributionListDemo")
            return ScalarDistributionListDemo()
        case "dev.outergroup.ScalarDistributionListDemoSingleCanvas":
            print("Creating ScalarDistributionListSingleCanvasDemo")
            return ScalarDistributionListSingleCanvasDemo()
        case "dev.outergroup.ScalarDistributionDemo":
            print("Creating ScalarDistributionDemo")
            return ScalarDistributionDemo()
        case "dev.outergroup.CheckerboardDemo":
            print("Creating CheckerboardDemo")
            return CheckerboardDemo()
        case "dev.outergroup.NumbersDemo":
            print("Creating NumbersDemo")
            return NumbersDemo()
        default:
            return nil
        }
    }

    // Creating new UUIDs per creation to avoid fragment ID collisions
    private static func getUniqueElementIDs() -> [String: UUID] {
        var ids = [String: UUID]()
        for i in 0..<N_WAVES {
            ids["sine\(i)"] = UUID()
        }
        return ids
    }

    // Creating new UUIDs per creation to avoid canvas ID collisions
    private static func getUniqueCanvasIDs() -> [String: UUID] {
        var ids = [String: UUID]()
        for i in 0..<N_WAVES {
            ids["progress\(i)"] = UUID()
        }
        return ids
    }
}
