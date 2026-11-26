import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

private let amber = Color(red:0.94, green:0.65, blue:0.34)
private let studioBackground = Color(red:0.10, green:0.105, blue:0.12)
private let panel = Color(red:0.145, green:0.15, blue:0.17)

// Observe the wheel only while it is over the preview. Other controls retain normal scrolling.
struct PreviewWheel: NSViewRepresentable {
    @Binding var zoom: Double
    func makeNSView(context: Context) -> WheelView { WheelView() }
    func updateNSView(_ view: WheelView, context: Context) { view.change = { delta in
        zoom = min(8, max(1, zoom * exp(delta * 0.025)))
    } }
    final class WheelView: NSView {
        var change: ((Double) -> Void)?
        var monitor: Any?
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let monitor { NSEvent.removeMonitor(monitor); self.monitor = nil }
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching:.scrollWheel) { [weak self] event in
                guard let self, event.window === self.window,
                      self.bounds.contains(self.convert(event.locationInWindow, from:nil)) else { return event }
                self.change?(Double(event.scrollingDeltaY))
                return nil
            }
        }
        override func hitTest(_ point:NSPoint) -> NSView? { nil }
        deinit { if let monitor { NSEvent.removeMonitor(monitor) } }
    }
}

// Drag the image inside its scroll view without affecting the surrounding controls.
struct PreviewPan: NSViewRepresentable {
    func makeNSView(context: Context) -> PanView { PanView() }
    func updateNSView(_ view: PanView, context: Context) {}

    final class PanView: NSView {
        private var startPoint: NSPoint?
        private var startOrigin: NSPoint?
        override var isFlipped: Bool { true }
        override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

        override func resetCursorRects() {
            addCursorRect(bounds, cursor: .openHand)
        }

        override func mouseDown(with event: NSEvent) {
            guard let clip = enclosingScrollView?.contentView else { return }
            startPoint = event.locationInWindow
            startOrigin = clip.bounds.origin
            NSCursor.closedHand.push()
        }

        override func mouseDragged(with event: NSEvent) {
            guard let startPoint, let startOrigin, let scroll = enclosingScrollView else { return }
            let clip = scroll.contentView
            let delta = NSPoint(x:event.locationInWindow.x - startPoint.x,
                                y:event.locationInWindow.y - startPoint.y)
            let proposed = NSPoint(x:startOrigin.x - delta.x,
                                   y:startOrigin.y + (clip.isFlipped ? delta.y : -delta.y))
            clip.scroll(to:clip.constrainBoundsRect(NSRect(origin:proposed, size:clip.bounds.size)).origin)
            scroll.reflectScrolledClipView(clip)
        }

        override func mouseUp(with event: NSEvent) {
            if startPoint != nil { NSCursor.pop() }
            startPoint = nil
            startOrigin = nil
        }
    }
}

struct Preview: View {
    let url: URL?
    @State private var image: NSImage?
    @State private var dimensions = ""
    @State private var previews: [URL: (NSImage, String)] = [:]
    @State private var zoom = 1.0
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius:12).fill(Color.black.opacity(0.27))
            if let image {
                GeometryReader { geometry in
                    ScrollViewReader { proxy in
                        ScrollView([.horizontal, .vertical]) {
                            Image(nsImage:image).resizable().scaledToFit()
                                .frame(width: max(1, geometry.size.width * zoom),
                                       height: max(1, geometry.size.height * zoom))
                                .overlay(PreviewPan())
                                .id("preview-image")
                        }
                        .defaultScrollAnchor(.center)
                        .onChange(of:zoom) {
                            DispatchQueue.main.async {
                                proxy.scrollTo("preview-image", anchor:.center)
                            }
                        }
                    }
                }.padding(14).background(PreviewWheel(zoom:$zoom))
                VStack {
                    HStack {
                        HStack {
                            Image(systemName:"minus.magnifyingglass")
                            Slider(value:$zoom,in:1...8).frame(width:100)
                            Image(systemName:"plus.magnifyingglass")
                            Text(String(format:"%.0f%%", zoom * 100)).monospacedDigit()
                            Button("100%") { zoom = 1 }.buttonStyle(.plain)
                        }.font(.caption).padding(7).background(.black.opacity(0.7),in:Capsule())
                        Spacer()
                        Text(dimensions).font(.caption.monospacedDigit())
                            .padding(7).background(.black.opacity(0.7),in:Capsule())
                    }.padding(10)
                    Spacer()
                }
            } else if url == nil {
                VStack(spacing:12) {
                    Image(nsImage:NSApp.applicationIconImage).resizable().frame(width:90,height:90)
                    Text("Drop images here").font(.title2.weight(.medium))
                    Text("One image, multiple images, or a folder.").foregroundStyle(.secondary)
                }
            } else { Text("Preview unavailable").foregroundStyle(.secondary) }
        }
        .task(id:url) {
            guard let url else { image = nil; return }
            if let cached = previews[url] {
                image = cached.0
                dimensions = cached.1
                return
            }
            let thumbnail = await Task.detached(priority:.userInitiated) { () -> (CGImage, String)? in
                guard let src = CGImageSourceCreateWithURL(url as CFURL,nil) else { return nil }
                let options: [CFString:Any] = [kCGImageSourceCreateThumbnailFromImageAlways:true,
                                                kCGImageSourceCreateThumbnailWithTransform:true,
                                                kCGImageSourceThumbnailMaxPixelSize:4096]
                guard let thumb = CGImageSourceCreateThumbnailAtIndex(src,0,options as CFDictionary) else { return nil }
                let meta = CGImageSourceCopyPropertiesAtIndex(src,0,nil) as? [CFString:Any]
                var w = meta?[kCGImagePropertyPixelWidth] as? Int ?? thumb.width
                var h = meta?[kCGImagePropertyPixelHeight] as? Int ?? thumb.height
                if (meta?[kCGImagePropertyOrientation] as? Int ?? 1) >= 5 { swap(&w,&h) }
                return (thumb,"\(w) × \(h)")
            }.value
            guard !Task.isCancelled, let thumbnail else { return }
            let loaded = NSImage(cgImage:thumbnail.0,size:.zero)
            previews[url] = (loaded, thumbnail.1)
            image = loaded
            dimensions = thumbnail.1
        }
    }
}

struct StudioWindow: View {
    @ObservedObject var studio: Studio
    @State private var over = false
    @State private var result = true
    @State private var showingOriginal = false
    private func caption(_ title:String) -> some View {
        Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
    }
    var body: some View {
        VStack(spacing:0) {
            HStack {
                VStack(alignment:.leading,spacing:2) {
                    Text("Kandinsky").font(.system(size:27,weight:.semibold,design:.rounded))
                    Text("ImageSR 2.0").font(.subheadline).foregroundStyle(.secondary)
                }
                Spacer()
                Button { studio.choose() } label:{ Label("Add…",systemImage:"plus") }
                    .keyboardShortcut("o",modifiers:.command)
            }.padding(.horizontal,22).padding(.vertical,14)
            Divider()
            HStack(alignment:.top,spacing:16) {
                VStack(alignment:.leading,spacing:10) {
                    Preview(url:result && !showingOriginal ? studio.current?.output ?? studio.selected : studio.selected)
                    if let item = studio.current {
                        HStack {
                            Text(item.id.lastPathComponent).font(.callout.weight(.medium)).lineLimit(1).truncationMode(.middle)
                            Spacer()
                            if item.output != nil {
                                Text("Hold to see original")
                                    .font(.caption)
                                    .padding(.horizontal,10).padding(.vertical,6)
                                    .background(Color.white.opacity(showingOriginal ? 0.18 : 0.08),in:RoundedRectangle(cornerRadius:6))
                                    .contentShape(Rectangle())
                                    .gesture(DragGesture(minimumDistance:0)
                                        .onChanged { _ in showingOriginal = true }
                                        .onEnded { _ in showingOriginal = false })
                                    .help("Hold to see original")
                                Picker("",selection:$result) { Text("Source").tag(false); Text("Result").tag(true) }
                                    .labelsHidden().pickerStyle(.segmented).frame(width:141)
                                Button { NSWorkspace.shared.activateFileViewerSelecting([item.output!]) } label:{
                                    Image(systemName:"folder")
                                }.help("Show output in Finder")
                            }
                        }
                        if !item.detail.isEmpty {
                            Text(item.detail).font(.caption).foregroundStyle(.orange).textSelection(.enabled)
                        }
                    }
                    HStack {
                        caption("Queue · \(studio.photos.count)")
                        Spacer()
                        if !studio.photos.isEmpty {
                            Button("Clear") { studio.photos.removeAll(); studio.selected = nil; studio.status = "Ready" }
                                .buttonStyle(.plain).font(.caption).disabled(studio.running)
                        }
                    }
                    List(selection:$studio.selected) {
                        ForEach(studio.photos) { pic in
                            HStack {
                                Image(systemName:pic.output != nil ? "checkmark.circle.fill" : (pic.status == "Error" ? "exclamationmark.circle":"photo"))
                                    .foregroundStyle(pic.output != nil ? Color.green : .secondary)
                                Text(pic.id.lastPathComponent).lineLimit(1).truncationMode(.middle)
                                Spacer()
                                Text(pic.status).font(.caption).foregroundStyle(.secondary)
                            }.tag(pic.id).help(pic.detail.isEmpty ? pic.id.path : pic.detail)
                            .contextMenu { Button("Remove") { studio.remove(pic.id) }.disabled(studio.running) }
                        }
                    }.listStyle(.plain).scrollContentBackground(.hidden).frame(height:105)
                        .background(panel,in:RoundedRectangle(cornerRadius:8))
                        .onDeleteCommand { if let url = studio.selected { studio.remove(url) } }
                }
                VStack(alignment:.leading,spacing:15) {
                    Text("Upscale").font(.headline)
                    Picker("Scale",selection:$studio.factor) { Text("×2").tag(2); Text("×4").tag(4) }
                        .labelsHidden().pickerStyle(.segmented)
                    HStack { Text("Creativity"); Spacer(); Text("\(studio.level) / 8").foregroundStyle(amber).monospacedDigit() }
                    Slider(value:Binding(get:{Double(studio.level)},set:{studio.level = Int($0)}),in:0...8,step:1)
                        .accessibilityLabel("Creativity")
                    HStack { Text("Faithful"); Spacer(); Text("Creative") }.font(.caption).foregroundStyle(.secondary)
                    Divider()
                    Text("Export").font(.headline)
                    Picker("Format",selection:$studio.format) { Text("PNG").tag("PNG"); Text("JPEG").tag("JPEG") }
                        .labelsHidden().pickerStyle(.segmented)
                    caption("Destination")
                    Menu {
                        Button("Next to originals") { studio.folder = nil }
                        Button("Choose a Folder…") { studio.chooseFolder() }
                    } label:{
                        Text(studio.folder?.lastPathComponent ?? "Next to originals").lineLimit(1)
                    }.help(studio.folder?.path ?? "Each original’s folder")
                    Spacer()
                    Label("MLX · Metal",systemImage:"cpu").font(.caption.monospaced()).foregroundStyle(.secondary)
                }.padding(16).frame(width:205).frame(maxHeight:.infinity)
                    .background(panel,in:RoundedRectangle(cornerRadius:12)).disabled(studio.running)
            }.padding(18)
            Divider()
            HStack(spacing:18) {
                VStack(alignment:.leading,spacing:6) {
                    Text(studio.status).lineLimit(2).font(.callout).help(studio.status).textSelection(.enabled)
                    if studio.running { ProgressView(value:studio.fraction).progressViewStyle(.linear) }
                }
                Spacer()
                if studio.running { Button("Stop", action:studio.cancel) }
                else {
                    Button { studio.run() } label:{
                        Label(studio.photos.count > 1 ? "Upscale Batch" : "Upscale",systemImage:"arrow.up.right")
                    }.buttonStyle(.borderedProminent).tint(amber).foregroundStyle(.black)
                        .disabled(studio.photos.isEmpty).keyboardShortcut(.return,modifiers:.command)
                }
            }.padding(.horizontal,20).frame(minHeight:58)
        }.background(studioBackground).preferredColorScheme(.dark).tint(amber)
            .frame(minWidth:720,minHeight:550)
            .overlay { if over { RoundedRectangle(cornerRadius:12).strokeBorder(amber,lineWidth:3).allowsHitTesting(false) } }
            .onDrop(of:[UTType.fileURL.identifier],isTargeted:$over) { providers in
                let group = DispatchGroup(), lock = NSLock()
                var urls = [URL]()
                for provider in providers {
                    group.enter()
                    provider.loadDataRepresentation(forTypeIdentifier:UTType.fileURL.identifier) { data,_ in
                        defer { group.leave() }
                        if let data, let url = URL(dataRepresentation:data,relativeTo:nil) {
                            lock.lock(); urls.append(url); lock.unlock()
                        }
                    }
                }
                group.notify(queue:.main) { studio.add(urls) }
                return true
            }
    }
}

@main
final class AppDelegate: NSObject, NSApplicationDelegate {
    let studio = Studio()
    var window: NSWindow!
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        withExtendedLifetime(delegate) { app.run() }
    }
    func applicationDidFinishLaunching(_ notification:Notification) {
        let menu = NSMenu(), appItem = NSMenuItem(), fileItem = NSMenuItem()
        menu.addItem(appItem); menu.addItem(fileItem)
        let appMenu = NSMenu(), fileMenu = NSMenu(title:"File")
        appItem.submenu = appMenu; fileItem.submenu = fileMenu
        appMenu.addItem(withTitle:"About Kandinsky ImageSR",action:#selector(NSApplication.orderFrontStandardAboutPanel(_:)),keyEquivalent:"")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle:"Quit Kandinsky ImageSR",action:#selector(NSApplication.terminate(_:)),keyEquivalent:"q")
        fileMenu.addItem(withTitle:"Add Images…",action:#selector(choose),keyEquivalent:"o").target = self
        fileMenu.addItem(withTitle:"Close",action:#selector(NSWindow.performClose(_:)),keyEquivalent:"w")
        NSApp.mainMenu = menu
        window = NSWindow(contentRect:NSRect(x:0,y:0,width:900,height:660),styleMask:[.titled,.closable,.miniaturizable,.resizable],
                          backing:.buffered,defer:false)
        window.title = "Kandinsky ImageSR"
        window.appearance = NSAppearance(named:.darkAqua)
        window.contentView = NSHostingView(rootView:StudioWindow(studio:studio))
        window.minSize = NSSize(width:740,height:600)
        window.center()
        window.setFrameAutosaveName("KandinskyStudio")
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps:true)
    }
    @objc func choose() { studio.choose() }
    func application(_ application:NSApplication,open urls:[URL]) {
        studio.add(urls)
        window?.makeKeyAndOrderFront(nil)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool { true }
    func applicationWillTerminate(_ notification:Notification) { studio.cancel() }
}
