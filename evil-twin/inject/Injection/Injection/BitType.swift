//
//  File.swift
//  
//
//  Created by paradiseduo on 2021/9/10.
//

import Foundation

public enum BitType {
    case x86
    case x64
    case x86Fat
    case x64Fat
    case none

    static func checkType(machoPath: String, header: fat_header, handle: (BitType, Bool) -> Void) {
        switch header.magic {
        case FAT_CIGAM, FAT_MAGIC:
            print("Please run 'lipo \(machoPath) -thin armv7 -output \(machoPath)_armv7' first")
            handle(.x86Fat, false)
        case FAT_CIGAM_64, FAT_MAGIC_64:
            print("Please run 'lipo \(machoPath) -thin armv64 -output \(machoPath)_arm64' first")
            handle(.x64Fat, false)
        case MH_MAGIC, MH_CIGAM:
            handle(.x86, header.magic == MH_CIGAM)
        case MH_MAGIC_64, MH_CIGAM_64:
            handle(.x64, header.magic == MH_CIGAM_64)
        default:
            print("Unkonw machO header")
            handle(.none, false)
        }
    }
}
