import SwiftUI

@MainActor
final class CardSession: ObservableObject {
    @Published var order: [StudyCard] = []
    @Published var position = 0
    @Published var flipped = false
    @Published var answers: [String: Bool] = [:]
    @Published var finished = false

    var current: StudyCard? { order.indices.contains(position) && !finished ? order[position] : nil }

    func rebuild(_ cards: [StudyCard], practice: Practice, shuffle: Bool, progress: ProgressStore) {
        var picked = cards
        switch practice {
        case .all: break
        case .notKnown: picked = picked.filter { !progress.known.contains($0.id) }
        case .review: picked = picked.filter { progress.review.contains($0.id) }
        }
        start(shuffle ? picked.shuffled() : picked)
    }

    func start(_ cards: [StudyCard]) {
        order = cards
        position = 0
        flipped = false
        answers = [:]
        finished = false
    }

    func move(_ delta: Int) {
        guard !order.isEmpty else { return }
        position = (position + delta + order.count) % order.count
        flipped = false
    }

    func answer(_ correct: Bool, progress: ProgressStore) {
        guard let card = current, flipped else { return }
        answers[card.id] = correct
        correct ? progress.markKnown(card.id) : progress.markReview(card.id)
        for step in 1...order.count {
            let p = (position + step) % order.count
            if answers[order[p].id] == nil {
                position = p
                flipped = false
                return
            }
        }
        finished = true
    }
}

struct CardsView: View {
    @EnvironmentObject private var progress: ProgressStore
    @ObservedObject var session: CardSession
    let rebuild: () -> Void
    let openLesson: (String) -> Void
    let setScope: (Scope) -> Void
    let setStudy: (Study) -> Void
    let showAll: () -> Void
    let study: Study
    let hasLesson: Bool
    let scope: Scope
    let practice: Practice

    var body: some View {
        VStack(spacing: 0) {
            if study == .test && session.finished {
                summary
            } else if let card = session.current {
                status
                cardView(card)
                Divider()
                actions(card)
            } else {
                empty
            }
        }
    }

    private var status: some View {
        let n = session.order.count
        let text: String
        if study == .test {
            let correct = session.answers.values.filter { $0 }.count
            text = "Card \(session.position + 1) of \(n) · Score \(correct)/\(session.answers.count)"
        } else {
            let known = session.order.filter { progress.known.contains($0.id) }.count
            let review = session.order.filter { progress.review.contains($0.id) }.count
            text = "Card \(session.position + 1) of \(n) · Known \(known) · Review \(review)"
        }
        return HStack {
            Text(text).monospacedDigit()
            Spacer()
            if let card = session.current {
                if progress.known.contains(card.id) { badge("Known", .green) }
                if progress.review.contains(card.id) { badge("Review", .orange) }
            }
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
    }

    private func badge(_ text: String, _ color: Color) -> some View {
        Text(text).font(.caption.bold()).foregroundStyle(color)
            .padding(.horizontal, 8).padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
    }

    private func cardView(_ item: StudyCard) -> some View {
        let showAnswer = study == .learn || session.flipped
        let html = "<div class=\"eyebrow\"><a href=\"#/lesson/\(item.lesson.id)\">\(item.lesson.title)</a></div>"
            + "<div class=\"question\">\(item.card.qHtml)</div>"
            + (showAnswer ? "<div class=\"answer\">\(item.card.aHtml)</div>"
                          : "<div class=\"hint\">Recall the answer, then press Space</div>")
        return HTMLView(html: html, openLesson: openLesson)
            .background(RoundedRectangle(cornerRadius: 16).fill(Color(nsColor: .textBackgroundColor)))
            .overlay(RoundedRectangle(cornerRadius: 16)
                .strokeBorder(showAnswer ? Color.accentColor.opacity(0.5) : Color.secondary.opacity(0.25)))
            .padding(.horizontal, 20)
            .padding(.bottom, 14)
    }

    private func actions(_ item: StudyCard) -> some View {
        HStack(spacing: 10) {
            Button { session.move(-1) } label: { Image(systemName: "chevron.left").frame(width: 28) }
                .keyboardShortcut(.leftArrow, modifiers: [])
            Spacer()
            if study == .learn {
                Button(progress.known.contains(item.id) ? "✓ Known" : "Known") { progress.toggleKnown(item.id) }
                    .tint(progress.known.contains(item.id) ? .green : nil)
                    .keyboardShortcut("k", modifiers: [])
                Button(progress.review.contains(item.id) ? "⚑ Review" : "Review") { progress.toggleReview(item.id) }
                    .tint(progress.review.contains(item.id) ? .orange : nil)
                    .keyboardShortcut("r", modifiers: [])
            } else if !session.flipped {
                Button("Show answer") { session.flipped = true }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.space, modifiers: [])
            } else {
                Button("✗ Missed it") { session.answer(false, progress: progress) }
                    .tint(.orange)
                    .keyboardShortcut("m", modifiers: [])
                Button("✓ Got it") { session.answer(true, progress: progress) }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .keyboardShortcut("g", modifiers: [])
            }
            Spacer()
            Button { session.move(1) } label: { Image(systemName: "chevron.right").frame(width: 28) }
                .keyboardShortcut(.rightArrow, modifiers: [])
        }
        .controlSize(.large)
        .padding(12)
    }

    private var empty: some View {
        let message: String
        switch (practice, hasLesson && scope == .lesson) {
        case (.review, _): message = "No cards here are marked for review."
        case (.notKnown, _): message = "Every card here is marked known."
        case (.all, true): message = "This lesson has no flash cards. Try the cards for its whole track."
        default: message = "Choose a lesson in the sidebar, or study the whole course."
        }
        return ContentUnavailableView {
            Label("No cards to practice", systemImage: "rectangle.on.rectangle.slash")
        } description: {
            Text(message)
        } actions: {
            if hasLesson && scope == .lesson {
                Button("Cards for this track") { setScope(.track) }.buttonStyle(.borderedProminent)
            }
            if practice != .all { Button("Show all cards", action: showAll) }
            Button("Whole course") { setScope(.all) }
        }
    }

    private var summary: some View {
        let correct = session.answers.values.filter { $0 }.count
        let missed = session.order.filter { session.answers[$0.id] == false }
        let total = session.order.count
        return ContentUnavailableView {
            Label("Test complete", systemImage: missed.isEmpty ? "star.fill" : "checkmark.seal")
        } description: {
            Text("\(correct) of \(total) correct (\(total == 0 ? 0 : correct * 100 / total)%). "
                 + (missed.isEmpty ? "Every card is now marked known."
                    : "Cards you got are marked known; the \(missed.count) you missed are marked for review."))
        } actions: {
            if !missed.isEmpty {
                Button("Retest the \(missed.count) missed") { session.start(missed) }.buttonStyle(.borderedProminent)
            }
            Button("Start a new test", action: rebuild)
            Button("Switch to Learn") { setStudy(.learn) }
        }
    }
}
