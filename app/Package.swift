// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "LLMServingLabs",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "LLMServingLabs")
    ]
)
