// Draws the app icon (a GPU chip marked "LLM") and writes an .iconset folder for iconutil,
// or with --web the PNGs for the web app (icon-192/512 plus a full-bleed maskable-512 for Android).
// Usage: swift scripts/make_icon.swift OUT.iconset
//        swift scripts/make_icon.swift --web OUT_DIR
import AppKit

enum Style { case mac, web, maskable }

let web = CommandLine.arguments[1] == "--web"
let out = URL(fileURLWithPath: CommandLine.arguments[web ? 2 : 1], isDirectory: true)
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

func render(_ px: Int, _ style: Style = .mac) -> Data {
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8,
        samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let s = CGFloat(px)

    // macOS icon grid: the tile is inset ~10% with a ~22% corner radius. Android crops
    // maskable icons to its own shape, keeping only the central 80% circle.
    let inset: CGFloat = style == .mac ? 0.1 : style == .web ? 0.02 : 0
    let tile = NSRect(x: s * inset, y: s * inset, width: s * (1 - 2 * inset), height: s * (1 - 2 * inset))
    let radius = style == .maskable ? 0 : tile.width * 0.225
    NSGradient(colors: [
        NSColor(red: 0.48, green: 0.42, blue: 0.96, alpha: 1),
        NSColor(red: 0.20, green: 0.27, blue: 0.72, alpha: 1),
    ])!.draw(in: NSBezierPath(roundedRect: tile, xRadius: radius, yRadius: radius), angle: -60)

    let chipSide = s * (style == .maskable ? 0.40 : tile.width / s * 0.50)
    let chip = NSRect(x: tile.midX - chipSide / 2, y: tile.midY - chipSide / 2, width: chipSide, height: chipSide)
    let white = NSColor(red: 0.97, green: 0.97, blue: 1.0, alpha: 1)
    white.setStroke()
    white.setFill()

    let pinLength = chipSide * 0.13
    let pinWidth = chipSide * 0.07
    for i in 0..<4 {
        let offset = chip.minX + chipSide * (0.2 + 0.2 * CGFloat(i)) - pinWidth / 2
        let vertical = chip.minY + chipSide * (0.2 + 0.2 * CGFloat(i)) - pinWidth / 2
        for rect in [
            NSRect(x: offset, y: chip.maxY, width: pinWidth, height: pinLength),
            NSRect(x: offset, y: chip.minY - pinLength, width: pinWidth, height: pinLength),
            NSRect(x: chip.minX - pinLength, y: vertical, width: pinLength, height: pinWidth),
            NSRect(x: chip.maxX, y: vertical, width: pinLength, height: pinWidth),
        ] {
            NSBezierPath(roundedRect: rect, xRadius: pinWidth / 2, yRadius: pinWidth / 2).fill()
        }
    }
    let body = NSBezierPath(roundedRect: chip, xRadius: chipSide * 0.12, yRadius: chipSide * 0.12)
    body.lineWidth = chipSide * 0.07
    body.stroke()

    let size = chipSide * 0.34
    let font = NSFont.systemFont(ofSize: size, weight: .heavy)
    let text = NSAttributedString(string: "LLM", attributes: [.font: font, .foregroundColor: white])
    let line = CTLineCreateWithAttributedString(text)
    let glyph = CTLineGetBoundsWithOptions(line, .useGlyphPathBounds)
    let ctx = NSGraphicsContext.current!.cgContext
    ctx.textPosition = CGPoint(x: chip.midX - glyph.midX, y: chip.midY - glyph.midY)
    CTLineDraw(line, ctx)

    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

if web {
    try render(192, .web).write(to: out.appendingPathComponent("icon-192.png"))
    try render(512, .web).write(to: out.appendingPathComponent("icon-512.png"))
    try render(512, .maskable).write(to: out.appendingPathComponent("maskable-512.png"))
} else {
    for size in [16, 32, 128, 256, 512] {
        try render(size).write(to: out.appendingPathComponent("icon_\(size)x\(size).png"))
        try render(size * 2).write(to: out.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
    }
}
