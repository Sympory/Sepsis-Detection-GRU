/**
 * SepsisAI — Giriş sayfası
 * Kullanıcı adı/şifre ile oturum açar; oturum HttpOnly çerezde tutulur.
 */

const form = document.getElementById('login-form');
const errorBox = document.getElementById('login-error');
const button = document.getElementById('login-btn');

if (new URLSearchParams(location.search).has('expired')) {
    const notice = document.getElementById('login-notice');
    notice.textContent = 'Oturumunuz sona erdi (30 dakika işlem yapılmadı). Lütfen tekrar giriş yapın.';
    notice.hidden = false;
}

form.addEventListener('submit', async event => {
    event.preventDefault();
    errorBox.hidden = true;
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;
    if (!username || !password) {
        return showError('Kullanıcı adı ve şifre gerekli');
    }
    button.disabled = true;
    button.textContent = 'Giriş yapılıyor…';
    try {
        const r = await fetch('/api/auth/login', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const d = await r.json();
        if (!d.success) {
            showError(d.error || 'Giriş başarısız');
            document.getElementById('password').select();
            return;
        }
        window.location.href = '/';
    } catch (e) {
        showError('Sunucuya ulaşılamıyor. Uygulama penceresinin açık olduğundan emin olun.');
    } finally {
        button.disabled = false;
        button.textContent = 'Giriş yap';
    }
});

function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
}

async function loadDemoAccounts() {
    try {
        const d = await (await fetch('/api/auth/demo-accounts')).json();
        if (!d.accounts || !d.accounts.length) return;
        document.getElementById('demo-list').innerHTML = d.accounts.map(a => `
            <button type="button" class="demo-account" data-u="${a.username}" data-p="${a.password}">
                <span class="role-badge">${a.role_label}</span>
                <span class="num">${a.username}</span>
                <span class="num muted">${a.password}</span>
            </button>`).join('');
        document.querySelectorAll('.demo-account').forEach(btn => btn.addEventListener('click', () => {
            document.getElementById('username').value = btn.dataset.u;
            document.getElementById('password').value = btn.dataset.p;
            document.getElementById('password').focus();
        }));
        document.getElementById('demo-accounts').hidden = false;
    } catch (e) { }
}

loadDemoAccounts();
