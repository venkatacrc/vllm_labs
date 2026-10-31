'use strict';

const $ = (sel) => document.querySelector(sel);

function h(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === 'class') el.className = v;
        else if (k === 'html') el.innerHTML = v;
        else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
        else if (typeof v === 'boolean') el[k] = v;
        else el.setAttribute(k, v);
    }
    for (const c of children.flat()) {
        if (c == null || c === false) continue;
        el.append(c instanceof Node ? c : String(c));
    }
    return el;
}

// ---- Saved state (localStorage) ----

const saved = {
    get(key, fallback) {
        try {
            const v = localStorage.getItem('llmLabs.' + key);
            return v == null ? fallback : JSON.parse(v);
        } catch {
            return fallback;
        }
    },
    set(key, value) {
        localStorage.setItem('llmLabs.' + key, JSON.stringify(value));
    },
};

const prefs = {
    mode: saved.get('mode', 'read'),
    study: saved.get('study', 'learn'),
    lesson: saved.get('lesson', null),
    scope: saved.get('scope', 'lesson'),
    practice: saved.get('practice', 'all'),
    shuffle: saved.get('shuffle', false),
    collapsed: saved.get('collapsed', []),
};

function setPref(key, value) {
    prefs[key] = value;
    saved.set(key, value);
}

const progress = {
    known: new Set(saved.get('known', [])),
    review: new Set(saved.get('review', [])),
    done: new Set(saved.get('done', [])),
    save() {
        saved.set('known', [...this.known]);
        saved.set('review', [...this.review]);
        saved.set('done', [...this.done]);
    },
    markKnown(id) { this.known.add(id); this.review.delete(id); this.save(); },
    markReview(id) { this.review.add(id); this.known.delete(id); this.save(); },
    toggleKnown(id) { this.known.has(id) ? this.known.delete(id) : this.markKnown(id); this.save(); },
    toggleReview(id) { this.review.has(id) ? this.review.delete(id) : this.markReview(id); this.save(); },
    toggleDone(id) { this.done.has(id) ? this.done.delete(id) : this.done.add(id); this.save(); },
};

let course = null;
let loadError = null;
/** Flat lists built once the course loads. */
let lessons = [];
let cards = [];
const lessonById = new Map();
const trackOf = new Map();
let search = '';

const session = { order: [], position: 0, flipped: false, answers: {}, finished: false };

// ---- Course helpers ----

function indexCourse(data) {
    course = data;
    lessons = [];
    cards = [];
    for (const track of data.tracks) {
        for (const lesson of track.lessons) {
            lessons.push(lesson);
            lessonById.set(lesson.id, lesson);
            trackOf.set(lesson.id, track);
            for (const card of lesson.cards) cards.push({ ...card, lesson });
        }
    }
}

const currentLesson = () => (prefs.lesson && lessonById.get(prefs.lesson)) || null;

function lessonText(lesson) {
    if (lesson.text == null) {
        lesson.text = (lesson.title + ' ' + lesson.html.replace(/<[^>]+>/g, ' ')).replace(/\s+/g, ' ').toLowerCase();
    }
    return lesson.text;
}

function cardsInScope() {
    const lesson = currentLesson();
    if (prefs.scope === 'lesson' && lesson) return cards.filter((c) => c.lesson === lesson);
    if (prefs.scope === 'track' && lesson) {
        const track = trackOf.get(lesson.id);
        return cards.filter((c) => trackOf.get(c.lesson.id) === track);
    }
    return cards;
}

function scopeLabel() {
    const lesson = currentLesson();
    if (!lesson || prefs.scope === 'all') return 'whole course';
    return prefs.scope === 'track' ? trackOf.get(lesson.id).title : 'this lesson';
}

function shuffled(list) {
    const a = [...list];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

function currentCard() {
    const c = session.order[session.position];
    return c && !session.finished ? c : null;
}

// ---- Navigation ----

function route() {
    if (!course) return;
    const m = location.hash.match(/^#\/lesson\/([\w.-]+)/);
    const id = m && lessonById.has(m[1]) ? m[1] : null;
    if (id !== prefs.lesson || prefs.mode !== 'read' || !id) {
        setPref('lesson', id);
        if (prefs.mode !== 'read') setPref('mode', 'read');
        rebuild(false);
    }
    render();
    const page = $('#stage .page');
    if (page) page.scrollTop = 0;
}

function openLesson(id) {
    const target = id ? `#/lesson/${id}` : '#/';
    if (location.hash === target) route();
    else location.hash = target;
}

function moveLesson(delta) {
    const lesson = currentLesson();
    const i = lesson ? lessons.indexOf(lesson) : -1;
    const next = lessons[i + delta];
    if (next) openLesson(next.id);
    else if (i + delta < 0) openLesson(null);
}

function studyCards(scope) {
    if (scope) setPref('scope', scope);
    setPref('mode', 'cards');
    rebuild();
}

// ---- Card session ----

function rebuild(draw = true) {
    let picked = cardsInScope();
    if (prefs.practice === 'notKnown') picked = picked.filter((c) => !progress.known.has(c.id));
    if (prefs.practice === 'review') picked = picked.filter((c) => progress.review.has(c.id));
    Object.assign(session, {
        order: prefs.shuffle ? shuffled(picked) : picked, position: 0, flipped: false, answers: {}, finished: false,
    });
    if (draw) render();
}

function startSession(order) {
    Object.assign(session, { order, position: 0, flipped: false, answers: {}, finished: false });
    render();
}

function move(delta) {
    const n = session.order.length;
    if (!n) return;
    session.position = (session.position + delta + n) % n;
    session.flipped = false;
    render();
}

function reveal() {
    if (prefs.study !== 'test' || session.flipped) return;
    session.flipped = true;
    render();
}

function answer(correct) {
    const card = currentCard();
    if (!card || !session.flipped) return;
    session.answers[card.id] = correct;
    correct ? progress.markKnown(card.id) : progress.markReview(card.id);
    const n = session.order.length;
    for (let step = 1; step <= n; step++) {
        const p = (session.position + step) % n;
        if (!(session.order[p].id in session.answers)) {
            session.position = p;
            session.flipped = false;
            render();
            return;
        }
    }
    session.finished = true;
    render();
}

// ---- Rendering ----

function render() {
    renderSwitches();
    renderStatus();
    renderStage();
    renderActions();
    if ($('#drawer').classList.contains('open')) renderDrawer();
}

function renderSwitches() {
    for (const [id, value] of [['#mode-switch', prefs.mode], ['#study-switch', prefs.study]]) {
        for (const b of $(id).querySelectorAll('button')) b.classList.toggle('on', b.dataset.value === value);
    }
    $('#study-switch').hidden = prefs.mode !== 'cards';
}

function renderStatus() {
    const el = $('#status');
    el.replaceChildren();
    if (!course) return;
    const lesson = currentLesson();
    if (prefs.mode === 'read') {
        if (!lesson) {
            el.append(h('span', {}, `${progress.done.size} of ${lessons.length} lessons read`),
                h('span', {}, `${cards.filter((c) => progress.known.has(c.id)).length} of ${cards.length} cards known`));
        } else {
            const track = trackOf.get(lesson.id);
            el.append(h('span', { class: 'crumb' }, h('strong', {}, track.title), ` · ${track.lessons.indexOf(lesson) + 1} of ${track.lessons.length}`));
            if (lesson.cards.length) el.append(h('span', {}, `${lesson.cards.length} cards`));
        }
        return;
    }
    const scope = h('select', { class: 'scope', 'aria-label': 'Cards from', onchange: (e) => { setPref('scope', e.target.value); rebuild(); } },
        [['lesson', 'This lesson'], ['track', 'This track'], ['all', 'Whole course']]
            .filter(([v]) => v === 'all' || lesson)
            .map(([v, t]) => h('option', { value: v, selected: (lesson ? prefs.scope : 'all') === v }, t)));
    el.append(scope);
    if (!session.order.length || (prefs.study === 'test' && session.finished)) return;
    const n = session.order.length;
    if (prefs.study === 'test') {
        const answered = Object.keys(session.answers).length;
        const correct = Object.values(session.answers).filter(Boolean).length;
        el.append(h('span', {}, `Card ${session.position + 1} of ${n} · Score ${correct}/${answered}`));
    } else {
        const known = session.order.filter((c) => progress.known.has(c.id)).length;
        const review = session.order.filter((c) => progress.review.has(c.id)).length;
        el.append(h('span', {}, h('strong', {}, `Card ${session.position + 1}`), ` of ${n} · Known ${known} · Review ${review}`));
    }
}

function renderStage() {
    const stage = $('#stage');
    stage.replaceChildren();
    if (loadError) {
        stage.append(h('div', { class: 'empty' }, h('h2', {}, 'Could not load the course'), h('p', {}, loadError)));
        return;
    }
    if (!course) {
        stage.append(h('div', { class: 'empty' }, 'Loading course…'));
        return;
    }
    if (prefs.mode === 'read') {
        stage.append(currentLesson() ? lessonPage(currentLesson()) : homePage());
        return;
    }
    if (prefs.study === 'test' && session.finished) {
        stage.append(testSummary());
        return;
    }
    const card = currentCard();
    if (!card) {
        stage.append(emptyCards());
        return;
    }
    const el = cardView(card, prefs.study === 'learn' || session.flipped);
    el.addEventListener('click', (e) => { if (!e.target.closest('button, a')) reveal(); });
    addSwipe(el);
    stage.append(el);
}

function fixLinks(root) {
    for (const a of root.querySelectorAll('a[href^="http"]')) {
        a.target = '_blank';
        a.rel = 'noopener';
    }
    return root;
}

function homePage() {
    return h('article', { class: 'page home' },
        h('h1', {}, course.title),
        h('p', { class: 'lead' }, 'From one GPU to a global service: systems foundations, vLLM, SGLang and llm-d labs on an 8xB200 node, and cloud deployment. Read a lesson, then test yourself with its cards.'),
        course.tracks.map((track) => {
            const done = track.lessons.filter((l) => progress.done.has(l.id)).length;
            const trackCards = track.lessons.flatMap((l) => l.cards);
            const known = trackCards.filter((c) => progress.known.has(c.id)).length;
            return h('section', { class: 'track' },
                h('div', { class: 'track-head' },
                    h('h2', {}, track.title),
                    h('span', { class: 'muted' }, `${done}/${track.lessons.length} read` + (trackCards.length ? ` · ${known}/${trackCards.length} known` : ''))),
                h('p', { class: 'muted' }, track.blurb),
                h('ol', { class: 'lesson-list' }, track.lessons.map((l) =>
                    h('li', {}, h('a', { href: `#/lesson/${l.id}`, class: progress.done.has(l.id) ? 'done' : '' },
                        h('span', { class: 'tick' }, progress.done.has(l.id) ? '✓' : ''), l.title)))));
        }),
        h('p', { class: 'muted foot' }, 'Source: ', h('a', { href: course.repo, target: '_blank', rel: 'noopener' }, course.repo.replace('https://', ''))));
}

function lessonPage(lesson) {
    const track = trackOf.get(lesson.id);
    const content = fixLinks(h('div', { class: 'content', html: lesson.html }));
    return h('article', { class: 'page lesson' },
        h('div', { class: 'eyebrow' }, track.title),
        h('h1', {}, lesson.title),
        content,
        h('div', { class: 'lesson-foot' },
            lesson.cards.length > 0 && h('button', { class: 'btn primary', onclick: () => studyCards('lesson') },
                `Study the ${lesson.cards.length} cards for this lesson`),
            h('button', { class: 'btn' + (progress.done.has(lesson.id) ? ' on-known' : ''), onclick: () => { progress.toggleDone(lesson.id); render(); } },
                progress.done.has(lesson.id) ? '✓ Read' : 'Mark as read'),
            h('a', { class: 'source', href: lesson.url, target: '_blank', rel: 'noopener' }, `View ${lesson.source} on GitHub`)));
}

function badges(card) {
    return h('div', { class: 'badges' },
        h('button', { class: 'label', type: 'button', onclick: () => openLesson(card.lesson.id) }, card.lesson.title),
        progress.known.has(card.id) && h('span', { class: 'badge known' }, 'Known'),
        progress.review.has(card.id) && h('span', { class: 'badge review' }, 'Review'));
}

function cardView(card, flipped) {
    return h('article', { class: 'card' + (flipped ? ' flipped' : '') },
        badges(card),
        h('div', { class: 'question', html: card.qHtml }),
        flipped
            ? fixLinks(h('div', { class: 'answer content', html: card.aHtml }))
            : h('div', { class: 'hint' }, 'Recall the answer, then tap the card'));
}

function emptyCards() {
    const lesson = currentLesson();
    const inScope = cardsInScope().length;
    let hint = 'Tap a lesson in the menu, or switch to the whole course.';
    if (inScope && prefs.practice === 'review') hint = `No cards from ${scopeLabel()} are marked for review.`;
    else if (inScope && prefs.practice === 'notKnown') hint = `Every card from ${scopeLabel()} is marked known.`;
    else if (lesson && prefs.scope === 'lesson') hint = 'This lesson has no flash cards. Try the cards for its whole track.';
    return h('div', { class: 'empty' },
        h('div', { class: 'big' }, '🗂'),
        h('h2', {}, 'No cards to practice'),
        h('p', {}, hint),
        h('div', { class: 'buttons' },
            lesson && prefs.scope === 'lesson' && h('button', { class: 'btn primary', onclick: () => studyCards('track') }, 'Cards for this track'),
            prefs.practice !== 'all' && h('button', { class: 'btn', onclick: () => { setPref('practice', 'all'); rebuild(); } }, 'Show all cards'),
            h('button', { class: 'btn', onclick: () => studyCards('all') }, 'Whole course')));
}

function testSummary() {
    const correct = Object.values(session.answers).filter(Boolean).length;
    const missed = session.order.filter((c) => session.answers[c.id] === false);
    const total = session.order.length;
    const percent = total ? Math.round((correct / total) * 100) : 0;
    return h('div', { class: 'summary' },
        h('div', { class: 'big' }, missed.length ? '✅' : '⭐'),
        h('h2', {}, 'Test complete'),
        h('div', { class: 'score' }, `${correct} of ${total} correct (${percent}%)`),
        h('p', {}, missed.length
            ? `Cards you got are marked known; the ${missed.length} you missed are marked for review.`
            : 'Every card is now marked known.'),
        h('div', { class: 'buttons' },
            missed.length > 0 && h('button', {
                class: 'btn primary',
                onclick: () => startSession(prefs.shuffle ? shuffled(missed) : missed),
            }, `Retest the ${missed.length} missed`),
            h('button', { class: 'btn', onclick: () => rebuild() }, 'Start a new test'),
            h('button', { class: 'btn', onclick: () => { setPref('study', 'learn'); rebuild(); } }, 'Switch to Learn')));
}

function renderActions() {
    const bar = $('#actions');
    bar.replaceChildren();
    if (!course) return;
    if (prefs.mode === 'read') {
        const lesson = currentLesson();
        const i = lesson ? lessons.indexOf(lesson) : -1;
        bar.append(...[
            h('button', { class: 'btn nav', onclick: () => moveLesson(-1), disabled: !lesson, 'aria-label': 'Previous lesson' }, '‹'),
            lesson
                ? h('button', { class: 'btn' + (progress.done.has(lesson.id) ? ' on-known' : ''), onclick: () => { progress.toggleDone(lesson.id); render(); } },
                    progress.done.has(lesson.id) ? '✓ Read' : 'Mark read')
                : h('button', { class: 'btn primary', onclick: () => openLesson((lessons.find((l) => !progress.done.has(l.id)) || lessons[0]).id) },
                    progress.done.size ? 'Continue' : 'Start the course'),
            lesson && lesson.cards.length > 0 && h('button', { class: 'btn', onclick: () => studyCards('lesson') }, `Cards (${lesson.cards.length})`),
            h('button', { class: 'btn nav', onclick: () => moveLesson(1), disabled: i >= lessons.length - 1, 'aria-label': 'Next lesson' }, '›'),
        ].filter(Boolean));
        return;
    }
    const card = currentCard();
    if (!card) return;
    const prev = h('button', { class: 'btn nav', onclick: () => move(-1), 'aria-label': 'Previous card' }, '‹');
    const next = h('button', { class: 'btn nav', onclick: () => move(1), 'aria-label': 'Next card' }, '›');
    if (prefs.study === 'learn') {
        bar.append(prev,
            h('button', {
                class: 'btn' + (progress.known.has(card.id) ? ' on-known' : ''),
                onclick: () => { progress.toggleKnown(card.id); render(); },
            }, progress.known.has(card.id) ? '✓ Known' : 'Known'),
            h('button', {
                class: 'btn' + (progress.review.has(card.id) ? ' on-review' : ''),
                onclick: () => { progress.toggleReview(card.id); render(); },
            }, progress.review.has(card.id) ? '⚑ Review' : 'Review'),
            next);
    } else if (!session.flipped) {
        bar.append(prev,
            h('button', { class: 'btn primary', onclick: reveal }, 'Show answer'),
            h('button', { class: 'btn nav', onclick: () => move(1), 'aria-label': 'Skip' }, '›'));
    } else {
        bar.append(prev,
            h('button', { class: 'btn bad', onclick: () => answer(false) }, '✗ Missed it'),
            h('button', { class: 'btn good', onclick: () => answer(true) }, '✓ Got it'),
            h('button', { class: 'btn nav', onclick: () => move(1), 'aria-label': 'Skip' }, '›'));
    }
}

function addSwipe(el) {
    let x0 = null, y0 = null;
    el.addEventListener('touchstart', (e) => {
        x0 = e.touches[0].clientX;
        y0 = e.touches[0].clientY;
    }, { passive: true });
    el.addEventListener('touchend', (e) => {
        if (x0 == null) return;
        const dx = e.changedTouches[0].clientX - x0;
        const dy = e.changedTouches[0].clientY - y0;
        x0 = null;
        if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy) * 1.5) move(dx < 0 ? 1 : -1);
    });
}

// ---- Drawer: lessons and options ----

function openDrawer() {
    renderDrawer();
    $('#drawer').classList.add('open');
    $('#drawer').setAttribute('aria-hidden', 'false');
    $('#backdrop').hidden = false;
}

function closeDrawer() {
    $('#drawer').classList.remove('open');
    $('#drawer').setAttribute('aria-hidden', 'true');
    $('#backdrop').hidden = true;
}

function checkRow(label, checked, onchange) {
    const id = 'c' + Math.random().toString(36).slice(2);
    return h('div', { class: 'row' },
        h('input', { type: 'checkbox', id, checked, onchange: (e) => onchange(e.target.checked) }),
        h('label', { for: id }, label));
}

function lessonRow(lesson, extra) {
    const current = lesson === currentLesson();
    return h('button', {
        class: 'lesson-row' + (current ? ' current' : ''),
        onclick: () => { closeDrawer(); openLesson(lesson.id); },
    },
        h('span', { class: 'tick' }, progress.done.has(lesson.id) ? '✓' : ''),
        h('span', { class: 'grow' }, lesson.title, extra && h('span', { class: 'snippet' }, extra)),
        lesson.cards.length > 0 && h('span', { class: 'count' }, lesson.cards.length));
}

function searchResults(query) {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    const hits = lessons.filter((l) => terms.every((t) => lessonText(l).includes(t)));
    const snippet = (l) => {
        const text = lessonText(l);
        const at = text.indexOf(terms[0]);
        return at < 0 ? '' : '…' + text.slice(Math.max(0, at - 40), at + 80) + '…';
    };
    return h('div', { class: 'panel' }, hits.length
        ? hits.slice(0, 40).map((l) => lessonRow(l, snippet(l)))
        : h('div', { class: 'row muted' }, 'No lessons match.'));
}

function renderDrawer() {
    const body = $('#drawer-body');
    const scroll = body.scrollTop;
    const focused = document.activeElement && document.activeElement.id === 'search';
    body.replaceChildren();
    if (!course) return;

    const input = h('input', {
        id: 'search', type: 'search', placeholder: 'Search lessons', value: search, autocomplete: 'off',
        oninput: (e) => { search = e.target.value; renderDrawer(); },
    });
    body.append(h('div', { class: 'search' }, input));
    if (search.trim()) {
        body.append(h('h4', {}, 'Results'), searchResults(search));
    } else {
        body.append(h('div', { class: 'panel' },
            h('button', { class: 'menu-btn', onclick: () => { closeDrawer(); openLesson(null); } }, '⌂ Contents and progress')));
        for (const track of course.tracks) {
            const collapsed = prefs.collapsed.includes(track.id);
            const done = track.lessons.filter((l) => progress.done.has(l.id)).length;
            body.append(h('div', { class: 'panel', style: 'margin-top:10px' },
                h('button', {
                    class: 'source-head' + (collapsed ? ' collapsed' : ''),
                    onclick: () => {
                        const set = new Set(prefs.collapsed);
                        collapsed ? set.delete(track.id) : set.add(track.id);
                        setPref('collapsed', [...set]);
                        renderDrawer();
                    },
                }, h('span', { class: 'chev' }, '▾'), track.title, h('span', { class: 'count' }, `${done}/${track.lessons.length}`)),
                !collapsed && track.lessons.map((l) => lessonRow(l))));
        }
    }

    const practice = h('select', { onchange: (e) => { setPref('practice', e.target.value); rebuild(); } },
        [['all', 'All cards'], ['notKnown', 'Not yet known'], ['review', 'Review only']]
            .map(([v, t]) => h('option', { value: v, selected: prefs.practice === v }, t)));
    body.append(h('h4', {}, 'Flash cards'), h('div', { class: 'panel' },
        h('div', { class: 'row' }, h('span', { class: 'grow' }, 'Practice'), practice),
        checkRow('Shuffle', prefs.shuffle, (on) => { setPref('shuffle', on); rebuild(); }),
        h('button', { class: 'menu-btn', onclick: () => { closeDrawer(); studyCards(); } }, '↺ Restart cards')));

    body.append(h('h4', {}, 'More'), h('div', { class: 'panel' },
        h('a', { class: 'menu-btn', href: course.repo, target: '_blank', rel: 'noopener' }, '⌥ Labs source code on GitHub'),
        h('button', {
            class: 'menu-btn danger',
            onclick: () => {
                if (!confirm('Clear every Known, Review and Read mark?')) return;
                progress.known.clear();
                progress.review.clear();
                progress.done.clear();
                progress.save();
                rebuild();
            },
        }, 'Reset progress')));

    const known = cards.filter((c) => progress.known.has(c.id)).length;
    $('#drawer-foot').textContent = `${progress.done.size}/${lessons.length} lessons read · ${known}/${cards.length} cards known`;
    body.scrollTop = scroll;
    if (focused) {
        input.focus();
        input.setSelectionRange(search.length, search.length);
    }
}

// ---- Wiring ----

$('#mode-switch').addEventListener('click', (e) => {
    const value = e.target.closest('button')?.dataset.value;
    if (!value || value === prefs.mode || !course) return;
    setPref('mode', value);
    if (value === 'cards') rebuild(); else render();
});

$('#study-switch').addEventListener('click', (e) => {
    const value = e.target.closest('button')?.dataset.value;
    if (!value || value === prefs.study || !course) return;
    setPref('study', value);
    rebuild();
});

$('#open-drawer').addEventListener('click', openDrawer);
$('#close-drawer').addEventListener('click', closeDrawer);
$('#backdrop').addEventListener('click', closeDrawer);
$('#go-home').addEventListener('click', () => openLesson(null));
window.addEventListener('hashchange', route);

document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'Escape') { closeDrawer(); return; }
    if (e.target.matches('input, textarea, select') || !course) return;
    if (prefs.mode === 'read') {
        const actions = { ArrowLeft: () => moveLesson(-1), ArrowRight: () => moveLesson(1) };
        if (actions[e.key]) { e.preventDefault(); actions[e.key](); }
        return;
    }
    const card = currentCard();
    if (!card) return;
    const test = prefs.study === 'test';
    const actions = {
        ArrowLeft: () => move(-1),
        ArrowRight: () => move(1),
        ' ': () => reveal(),
        g: () => test && answer(true),
        m: () => test && answer(false),
        k: () => { if (!test) { progress.toggleKnown(card.id); render(); } },
        r: () => { if (!test) { progress.toggleReview(card.id); render(); } },
    };
    const run = actions[e.key.length === 1 ? e.key.toLowerCase() : e.key];
    if (run) {
        e.preventDefault();
        run();
    }
});

render();

fetch('course.json')
    .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
    })
    .then((data) => {
        indexCourse(data);
        if (!location.hash && prefs.lesson && lessonById.has(prefs.lesson)) {
            history.replaceState(null, '', `#/lesson/${prefs.lesson}`);
        }
        const mode = prefs.mode;
        route();
        if (mode === 'cards' && location.hash.startsWith('#/lesson/')) studyCards();
    })
    .catch((err) => {
        loadError = `Open this page once while online so the course can be saved for offline use. (${err.message})`;
        render();
    });

if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
}
