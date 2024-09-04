//
//  Inject.swift
//  Injection
//
//  Created by paradiseduo on 2022/12/6.
//

import Foundation

public struct Inject {
    public static func injectIPA(ipaPath: String,
                                 cmdType: LCType,
                                 injectPath: String,
                                 finishHandle: (Bool) -> Void) {
        var result = false
        var injectFilePath = "."
        if injectPath.hasPrefix("@") {
            let arr = injectPath.components(separatedBy: "/")
            for item in arr {
                if item.contains("@") {
                    continue
                } else {
                    injectFilePath += "/\(item)"
                }
            }
        } else {
            injectFilePath += "/\(injectPath)"
        }
        var iPath = ""
        var iName = ""
        var injectPathNew = ""
        let components = injectPath.components(separatedBy: "/")

        if injectPath.hasSuffix(".framework") {
            iName = injectFilePath.components(separatedBy: "/").last!
            iPath = injectFilePath

            let frameworkExeName = iName.components(separatedBy: ".").first!
            if components.count > 1 {
                if components.first!.hasPrefix("@") {
                    injectPathNew = "\(components.first!)/Inject/\(iName)/\(frameworkExeName)"
                } else {
                    injectPathNew = "@executable_path/Inject/\(iName)/\(frameworkExeName)"
                }
            } else {
                injectPathNew = "@executable_path/Inject/\(iName)/\(frameworkExeName)"
            }
        } else if injectPath.hasSuffix(".dylib") {
            iName = injectFilePath.components(separatedBy: "/").last!
            iPath = injectFilePath

            if components.count > 1 {
                if components.first!.hasPrefix("@") {
                    injectPathNew = "\(components.first!)/Inject/\(iName)"
                } else {
                    injectPathNew = "@executable_path/Inject/\(iName)"
                }
            } else {
                injectPathNew = "@executable_path/Inject/\(iName)"
            }
        } else if injectPath.contains(".framework") {
            let aaa = injectFilePath.components(separatedBy: "/")
            let bbb = aaa.dropLast()
            iName = bbb.last!
            for item in bbb {
                iPath += item+"/"
            }

            if components.count > 1 {
                if components.first!.hasPrefix("@") {
                    injectPathNew = "\(components.first!)/Inject/\(iName)/\(aaa.last!)"
                } else {
                    injectPathNew = "@executable_path/Inject/\(iName)/\(aaa.last!)"
                }
            } else {
                injectPathNew = "@executable_path/Inject/\(iName)/\(aaa.last!)"
            }
        }

        if injectPathNew.hasSuffix("/") {
            injectPathNew.removeLast()
        }

        if iPath == "" || iName == "" || !FileManager.default.fileExists(atPath: iPath) {
            print("Need a dylib or framework file to inject")
            finishHandle(result)
            return
        }

        let targetUrl = "."
        Shell.run("unzip -o \(ipaPath) -d \(targetUrl)") { status, output in
            if status == 0 {
                let payload = targetUrl+"/Payload"
                do {
                    let fileList = try FileManager.default.contentsOfDirectory(atPath: payload)
                    var machoPath = ""
                    var appPath = ""
                    for item in fileList where item.hasSuffix(".app") {
                        appPath = payload + "/\(item)"
                        machoPath = appPath+"/\(item.components(separatedBy: ".")[0])"
                        break
                    }

                    try FileManager.default.createDirectory(atPath: "\(appPath)/Inject/",
                                                            withIntermediateDirectories: true,
                                                            attributes: nil)
                    try FileManager.default.moveItem(atPath: iPath,
                                                     toPath: "\(appPath)/Inject/\(iName)")

                    injectMachO(machoPath: machoPath,
                                cmdType: cmdType,
                                backup: false,
                                injectPath: injectPathNew) { success in
                        if success {
                            Shell.run("zip -r \(ipaPath) \(payload)") { status, output in
                                if status == 0 {
                                    print("Inject \(injectPath) finish, new IPA file is \(ipaPath)")
                                    result = true
                                } else {
                                    print("\(output)")
                                }
                            }
                        }
                    }
                    try FileManager.default.removeItem(atPath: payload)
                } catch let error {
                    print("\(error)")
                }
            } else {
                print("\(output)")
            }
        }
        finishHandle(result)
    }

    public static func removeIPA(ipaPath: String,
                                 cmdType: LCType,
                                 injectPath: String,
                                 finishHandle: (Bool) -> Void) {
        var result = false
        let targetUrl = "."
        Shell.run("unzip -o \(ipaPath) -d \(targetUrl)") { status, output in
            if status == 0 {
                let payload = targetUrl+"/Payload"
                do {
                    let fileList = try FileManager.default.contentsOfDirectory(atPath: payload)
                    var machoPath = ""
                    var appPath = ""
                    for item in fileList where item.hasSuffix(".app") {
                        appPath = payload + "/\(item)"
                        machoPath = appPath+"/\(item.components(separatedBy: ".")[0])"
                        break
                    }
                    removeMachO(machoPath: machoPath,
                                cmdType: cmdType,
                                backup: false,
                                injectPath: injectPath) { success in
                        if success {
                            Shell.run("zip -r \(ipaPath) \(payload)") { status, output in
                                if status == 0 {
                                    print("Remove \(injectPath) finish, new IPA file is \(ipaPath)")
                                    result = true
                                } else {
                                    print("\(output)")
                                }
                            }
                        }
                    }
                    try FileManager.default.removeItem(atPath: payload)
                } catch let error {
                    print("\(error)")
                }
            } else {
                print("\(output)")
            }
        }
        finishHandle(result)
    }

    public static func removeMachO(machoPath: String,
                                   cmdType: LCType,
                                   backup: Bool,
                                   injectPath: String,
                                   finishHandle: (Bool) -> Void) {
        let cmdType = LCType.get(cmdType.rawValue)
        var result = false
        FileManager.open(machoPath: machoPath, backup: backup) { data in
            if let binary = data {
                let fatHeader = binary.extract(fat_header.self)
                BitType.checkType(machoPath: machoPath, header: fatHeader) { type, _ in
                    if injectPath.count > 0 {
                        LoadCommand.remove(binary: binary,
                                           dylibPath: injectPath,
                                           cmd: cmdType,
                                           type: type) { newBinary in
                            result = Inject.writeFile(newBinary: newBinary,
                                                      machoPath: machoPath,
                                                      successTitle: "Remove \(injectPath) Finish",
                                                      failTitle: "Remove \(injectPath) failed")
                        }
                    } else {
                        print("Need dylib to inject")
                    }
                }
            }
        }
        finishHandle(result)
    }

    public static func injectMachO(machoPath: String,
                                   cmdType: LCType,
                                   backup: Bool,
                                   injectPath: String,
                                   finishHandle: (Bool) -> Void) {
        let cmdType = LCType.get(cmdType.rawValue)
        var result = false
        FileManager.open(machoPath: machoPath, backup: backup) { data in
            if let binary = data {
                let fatHeader = binary.extract(fat_header.self)
                BitType.checkType(machoPath: machoPath, header: fatHeader) { type, isByteSwapped in
                    if injectPath.count > 0 {
                        LoadCommand.couldInjectLoadCommand(binary: binary,
                                                           dylibPath: injectPath,
                                                           type: type,
                                                           isByteSwapped: isByteSwapped) { canInject in
                            LoadCommand.inject(binary: binary,
                                               dylibPath: injectPath,
                                               cmd: cmdType,
                                               type: type,
                                               canInject: canInject) { newBinary in
                                result = Inject.writeFile(newBinary: newBinary,
                                                          machoPath: machoPath,
                                                          successTitle: "Inject \(injectPath) Finish",
                                                          failTitle: "Inject \(injectPath) failed")
                            }
                        }
                    } else {
                        print("Need dylib to inject")
                    }
                }
            }
        }
        finishHandle(result)
    }

    private static func writeFile(newBinary: Data?,
                                  machoPath: String,
                                  successTitle: String,
                                  failTitle: String) -> Bool {
        if let newBinary = newBinary {
            do {
                try newBinary.write(to: URL(fileURLWithPath: machoPath))
                print(successTitle)
                return true
            } catch let error {
                print(error)
            }
        }
        print(failTitle)
        return false
    }
}
