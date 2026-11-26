import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ImageIO

struct Picture: Identifiable {
    let id: URL
    var status = "Waiting"
    var detail = ""
    var output: URL? = nil
}

final class Studio: ObservableObject {
    @Published var photos: [Picture] = []
    @Published var selected: URL?
    @Published var running = false
    @Published var status = "Ready"
    @Published var fraction = 0.0
    @Published var factor = UserDefaults.standard.integer(forKey: "factor") == 4 ? 4 : 2 {
        didSet { UserDefaults.standard.set(factor, forKey:"factor") }
    }
    @Published var level = UserDefaults.standard.object(forKey: "level") as? Int ?? 4 {
        didSet { UserDefaults.standard.set(level, forKey:"level") }
    }
    @Published var format = UserDefaults.standard.string(forKey: "format") ?? "PNG" {
        didSet { UserDefaults.standard.set(format, forKey:"format") }
    }
    @Published var folder: URL?
    var current: Picture? { photos.first { $0.id == selected } }
    private var process: Process?
    private var stopping = false
    private let scanner = DispatchQueue(label:"studio.import", qos:.userInitiated)
    let types = ["png","jpg","jpeg","jfif","webp","heic","heif","avif","bmp","tif","tiff"]
    func add(_ urls: [URL]) {
        let allowed = Set(types)
        scanner.async {
            var found = [URL]()
            func consider(_ url: URL) {
                guard allowed.contains(url.pathExtension.lowercased()),
                      url.deletingPathExtension().lastPathComponent.range(of:"_KSR_x[24](-[0-9]+)?$", options:[.regularExpression,.caseInsensitive]) == nil else { return }
                found.append(url.standardizedFileURL)
            }
            for url in urls {
                if (try? url.resourceValues(forKeys:[.isDirectoryKey]).isDirectory) == true {
                    let listing = FileManager.default.enumerator(at:url, includingPropertiesForKeys:[.isRegularFileKey],
                                                                 options:[.skipsHiddenFiles,.skipsPackageDescendants])
                    while let file = listing?.nextObject() as? URL {
                        if (try? file.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile) == true { consider(file) }
                    }
                } else { consider(url) }
            }
            let sorted = found.sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
            DispatchQueue.main.async {
                var known = Set(self.photos.map(\.id))
                var firstAdded: URL?
                for url in sorted where known.insert(url).inserted {
                    self.photos.append(Picture(id:url))
                    if firstAdded == nil { firstAdded = url }
                }
                if let selection = firstAdded ?? sorted.first { self.selected = selection }
                if sorted.isEmpty && !self.running { self.status = "No supported images found." }
            }
        }
    }

    func choose() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = true
        panel.allowedContentTypes = types.compactMap { UTType(filenameExtension:$0) } + [.folder]
        panel.prompt = "Add"
        if panel.runModal() == .OK { add(panel.urls) }
    }
    func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = "Choose"
        if panel.runModal() == .OK { folder = panel.url }
    }
    func remove(_ url: URL) {
        guard !running, let i = photos.firstIndex(where:{ $0.id == url }) else { return }
        photos.remove(at:i)
        if selected == url { selected = photos.isEmpty ? nil : photos[min(i,photos.count-1)].id }
    }
    func cancel() {
        guard let process = process, process.isRunning else { return }
        stopping = true
        status = "Stopping…"
        process.terminate()
    }

    func run() {
        guard !running, !photos.isEmpty, let res = Bundle.main.resourceURL else { return }
        let jobs = photos.map(\.id)
        let p = Process(), stdout = Pipe(), stdin = Pipe()
        p.executableURL = res.appendingPathComponent("runtime/bin/python3")
        p.arguments = ["-B", res.appendingPathComponent("worker.py").path, res.appendingPathComponent("model").path]
        p.environment = ProcessInfo.processInfo.environment.merging(["PYTHONNOUSERSITE":"1","PYTHONUNBUFFERED":"1",
                                                                      "PYTHONPATH":"","LC_CTYPE":"UTF-8"]) { _,new in new }
        p.standardOutput = stdout
        p.standardInput = stdin
        let log = FileManager.default.temporaryDirectory.appendingPathComponent("Kandinsky-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath:log.path,contents:nil)
        let request: [String:Any] = ["factor":factor,"level":level,"format":format,"folder":folder?.path ?? "","files":jobs.map(\.path)]
        do {
            let input = try JSONSerialization.data(withJSONObject:request)
            let logHandle = try FileHandle(forWritingTo:log)
            p.standardError = logHandle
            try p.run()
            try? logHandle.close()
            process = p
            running = true
            stopping = false
            fraction = 0
            status = "Loading model · MLX / Metal"
            for i in photos.indices { photos[i] = Picture(id:photos[i].id) }
            DispatchQueue.global(qos:.userInitiated).async {
                try? stdin.fileHandleForWriting.write(contentsOf:input)
                try? stdin.fileHandleForWriting.close()
            }
            DispatchQueue.global(qos:.userInitiated).async {
                var pending = Data()
                let handle = stdout.fileHandleForReading
                while true {
                    let block = handle.availableData
                    if block.isEmpty { break }
                    pending.append(block)
                    while let end = pending.firstIndex(of:10) {
                        let line = pending[..<end]
                        if let event = (try? JSONSerialization.jsonObject(with:line)) as? [String:Any] {
                            DispatchQueue.main.async { self.update(event,jobs:jobs) }
                        }
                        pending.removeSubrange(...end)
                    }
                }
                p.waitUntilExit()
                DispatchQueue.main.async {
                    let completed = self.photos.filter { jobs.contains($0.id) && $0.output != nil }.count
                    let failed = self.photos.filter { jobs.contains($0.id) && $0.status == "Error" }.count
                    if self.stopping {
                        self.status = "Batch stopped · \(completed)/\(jobs.count) exported"
                    } else if p.terminationStatus != 0 || completed + failed != jobs.count {
                        let error = (try? String(contentsOf:log, encoding:.utf8))?.split(separator:"\n").last
                        self.status = error.map(String.init) ?? "The engine stopped (\(p.terminationStatus))."
                    } else {
                        self.status = "\(completed)/\(jobs.count) exported" + (failed > 0 ? " · \(failed) failed" : "")
                        self.fraction = 1
                    }
                    for i in self.photos.indices where self.photos[i].status.hasPrefix("Tile") {
                        self.photos[i].status = "Stopped"
                    }
                    self.process = nil
                    self.running = false
                    try? FileManager.default.removeItem(at:log)
                }
            }
        } catch {
            status = error.localizedDescription
            try? FileManager.default.removeItem(at:log)
        }
    }
    private func update(_ event:[String:Any],jobs:[URL]) {
        guard let index = event["index"] as? Int, jobs.indices.contains(index),
              let i = photos.firstIndex(where:{ $0.id == jobs[index] }) else { return }
        switch event["event"] as? String {
        case "tile":
            let tile = event["tile"] as? Int ?? 0, total = event["total"] as? Int ?? 1
            photos[i].status = "Tile \(tile+1)/\(total)"
            status = "Image \(index+1)/\(jobs.count) · \(photos[i].id.lastPathComponent) · \(photos[i].status)"
            fraction = (Double(index) + Double(tile)/Double(total))/Double(jobs.count)
        case "done":
            if let path = event["path"] as? String {
                photos[i].output = URL(fileURLWithPath:path)
                photos[i].status = String(format:"Exported · %.1f s", event["seconds"] as? Double ?? 0)
            }
        case "failed":
            photos[i].status = "Error"
            photos[i].detail = event["reason"] as? String ?? "Export failed."
        default: break
        }
    }
}
