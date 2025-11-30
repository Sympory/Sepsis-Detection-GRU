/**
 * Login Page JavaScript
 * Handles authentication, hospital selection, and form submission
 */

// API base URL
const API_BASE = window.location.origin;

// Load hospitals on page load
document.addEventListener('DOMContentLoaded', function () {
    loadHospitals();
    checkExistingSession();
});

/**
 * Load hospitals from API
 */
async function loadHospitals() {
    try {
        const response = await fetch(`${API_BASE}/api/hospitals`);
        const data = await response.json();

        if (data.success && data.hospitals) {
            const select = document.getElementById('hospital');

            data.hospitals.forEach(hospital => {
                const option = document.createElement('option');
                option.value = hospital.id;
                option.textContent = `${hospital.name} (${hospital.code})`;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Failed to load hospitals:', error);
        showError('Hastane listesi yüklenemedi. Lütfen sayfayı yenileyin.');
    }
}

/**
 * Check if user already has valid session
 */
async function checkExistingSession() {
    const sessionId = getCookie('session_id');

    if (sessionId) {
        try {
            const response = await fetch(`${API_BASE}/api/auth/me`, {
                headers: {
                    'X-Session-ID': sessionId
                }
            });

            if (response.ok) {
                // Valid session exists, redirect to dashboard
                window.location.href = '/index.html';
            }
        } catch (error) {
            // Invalid session, continue to login
            console.log('No valid session found');
        }
    }
}

/**
 * Handle login form submission
 */
document.getElementById('loginForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const hospital = document.getElementById('hospital').value;
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const rememberMe = document.getElementById('rememberMe').checked;

    // Validation
    if (!hospital) {
        showError('Lütfen bir hastane seçiniz');
        return;
    }

    if (!username || !password) {
        showError('Kullanıcı adı ve şifre gereklidir');
        return;
    }

    // Show loading state
    setLoading(true);
    hideError();

    try {
        const response = await fetch(`${API_BASE}/api/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: username,
                password: password,
                hospital_id: parseInt(hospital),
                remember_me: rememberMe
            })
        });

        const data = await response.json();

        if (response.ok && data.success) {
            // Login successful
            const sessionId = data.session_id;
            const maxAge = rememberMe ? 30 * 24 * 60 * 60 : null; // 30 days or session

            // Set session cookie
            setCookie('session_id', sessionId, maxAge);

            // Store user info
            localStorage.setItem('user', JSON.stringify(data.user));

            // Show success message
            showSuccess('Giriş başarılı! Yönlendiriliyorsunuz...');

            // Redirect to dashboard
            setTimeout(() => {
                window.location.href = '/index.html';
            }, 1000);

        } else {
            // Login failed
            showError(data.error || 'Giriş başarısız. Lütfen bilgilerinizi kontrol edin.');
            setLoading(false);
        }

    } catch (error) {
        console.error('Login error:', error);
        showError('Bir hata oluştu. Lütfen tekrar deneyin.');
        setLoading(false);
    }
});

/**
 * Toggle password visibility
 */
function togglePassword() {
    const passwordInput = document.getElementById('password');
    const eyeIcon = document.getElementById('eyeIcon');

    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        eyeIcon.textContent = '👁️‍🗨️';
    } else {
        passwordInput.type = 'password';
        eyeIcon.textContent = '👁️';
    }
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'flex';

    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hide error message
 */
function hideError() {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.style.display = 'none';
}

/**
 * Show success message
 */
function showSuccess(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'flex';
    errorDiv.style.background = '#d1fae5';
    errorDiv.style.borderColor = '#a7f3d0';
    errorDiv.style.color = '#065f46';
}

/**
 * Set loading state
 */
function setLoading(loading) {
    const button = document.getElementById('loginButton');
    const buttonText = document.getElementById('buttonText');
    const buttonLoader = document.getElementById('buttonLoader');

    if (loading) {
        button.disabled = true;
        buttonText.style.display = 'none';
        buttonLoader.style.display = 'inline-block';
    } else {
        button.disabled = false;
        buttonText.style.display = 'inline';
        buttonLoader.style.display = 'none';
    }
}

/**
 * Set cookie
 */
function setCookie(name, value, maxAge = null) {
    let cookie = `${name}=${value}; path=/; SameSite=Strict`;

    if (maxAge) {
        cookie += `; max-age=${maxAge}`;
    }

    // Use Secure flag in production (HTTPS)
    if (window.location.protocol === 'https:') {
        cookie += '; Secure';
    }

    document.cookie = cookie;
}

/**
 * Get cookie value
 */
function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);

    if (parts.length === 2) {
        return parts.pop().split(';').shift();
    }

    return null;
}

/**
 * Delete cookie
 */
function deleteCookie(name) {
    document.cookie = `${name}=; path=/; max-age=0`;
}

// Handle Enter key in password field
document.getElementById('password').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        document.getElementById('loginForm').dispatchEvent(new Event('submit'));
    }
});
