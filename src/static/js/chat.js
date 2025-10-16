// Ebla-RAG Chat UI JavaScript

// State management
let currentSessionId = null;
let retrievedDocuments = [];

// DOM Elements
document.addEventListener('DOMContentLoaded', () => {
    const newChatBtn = document.getElementById('new-chat-btn');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const messagesContainer = document.getElementById('messages-container');
    const sessionsContainer = document.getElementById('sessions-list');
    const contextContainer = document.getElementById('context-content');
    const loadingIndicator = document.getElementById('loading-indicator');

    // Event Listeners
    newChatBtn.addEventListener('click', createNewSession);
    messageForm.addEventListener('submit', sendMessage);

    // Load sessions on page load
    loadSessions();

    // Functions
    async function loadSessions() {
        try {
            // Fetch all available sessions from the API
            const response = await fetch('/api/v1/sessions');
            
            if (!response.ok) {
                throw new Error('Failed to load sessions');
            }
            
            const data = await response.json();
            
            // Clear sessions list
            sessionsContainer.innerHTML = '';
            
            // Add sessions to the UI
            if (data.sessions && data.sessions.length > 0) {
                data.sessions.forEach(session => {
                    addSessionToList(session);
                });
                
                // Load the first session if no current session
                const savedSessionId = localStorage.getItem('currentSessionId');
                if (savedSessionId) {
                    currentSessionId = savedSessionId;
                    await loadSessionHistory(currentSessionId);
                } else if (data.sessions.length > 0) {
                    currentSessionId = data.sessions[0].id;
                    localStorage.setItem('currentSessionId', currentSessionId);
                    await loadSessionHistory(currentSessionId);
                }
            }
        } catch (error) {
            console.error('Error loading sessions:', error);
        }
    }

    async function createNewSession() {
        try {
            // Clear UI first
            messagesContainer.innerHTML = '';
            contextContainer.innerHTML = '';
            document.getElementById('summary-container').innerHTML = '';
            
            // Create a new session via API
            const response = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: '',
                    create_only: true
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to create new session');
            }
            
            const data = await response.json();
            
            // Set the new session as current
            currentSessionId = data.session_id;
            localStorage.setItem('currentSessionId', currentSessionId);
            
            // Update UI to show we're in a new session
            await loadSessions();
            highlightCurrentSession();
            
        } catch (error) {
            console.error('Error creating new session:', error);
        }
    }

    async function loadSessionHistory(sessionId) {
        try {
            showLoading(true);
            const response = await fetch(`/api/v1/history/${sessionId}`);
            
            if (!response.ok) {
                throw new Error('Failed to load session history');
            }
            
            const data = await response.json();
            
            // Clear existing messages and summary
            messagesContainer.innerHTML = '';
            document.getElementById('summary-container').innerHTML = '';
            
            // Display summary if available
            if (data.summary) {
                const summaryContainer = document.getElementById('summary-container');
                summaryContainer.innerHTML = `
                    <div class="summary-header">
                        <h4>Conversation Summary</h4>
                    </div>
                    <div class="summary-content">
                        ${data.summary}
                    </div>
                `;
                summaryContainer.style.display = 'block';
            }
            
            // Add messages to the UI
            data.messages.forEach(msg => {
                addMessageToUI(msg.role, msg.content);
            });
            
            // Scroll to bottom
            scrollToBottom();
            
            // Highlight the current session without reloading all sessions
            highlightCurrentSession();
            
        } catch (error) {
            console.error('Error loading session history:', error);
        } finally {
            showLoading(false);
        }
    }

    async function sendMessage(event) {
        event.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Add user message to UI
        addMessageToUI('user', message);
        
        // Clear input
        messageInput.value = '';
        
        // Scroll to bottom
        scrollToBottom();
        
        try {
            showLoading(true);
            
            // Prepare request body
            const requestBody = {
                message: message,
                method: 'langchain'
            };
            
            // If we have a session ID, include it
            if (currentSessionId) {
                requestBody.session_id = currentSessionId;
            }
            
            // Send message to API
            const response = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });
            
            if (!response.ok) {
                throw new Error('Failed to send message');
            }
            
            const data = await response.json();
            
            // Save session ID if this is a new session
            if (!currentSessionId) {
                currentSessionId = data.session_id;
                localStorage.setItem('currentSessionId', currentSessionId);
                updateSessionsList();
            }
            
            // Add assistant response to UI
            addMessageToUI('assistant', data.answer);
            
            // Update retrieved documents
            updateRetrievedDocuments(data.retrieved_documents);
            
            // Scroll to bottom
            scrollToBottom();
            
        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToUI('assistant', 'Sorry, there was an error processing your request.');
        } finally {
            showLoading(false);
        }
    }

    function addMessageToUI(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        contentDiv.textContent = content;
        
        const metaDiv = document.createElement('div');
        metaDiv.className = 'meta';
        metaDiv.textContent = new Date().toLocaleTimeString();
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(metaDiv);
        
        messagesContainer.appendChild(messageDiv);
    }

    function updateRetrievedDocuments(documents) {
        // Clear existing documents
        contextContainer.innerHTML = '';
        
        if (!documents || documents.length === 0) {
            const noDocsDiv = document.createElement('div');
            noDocsDiv.textContent = 'No relevant documents found.';
            contextContainer.appendChild(noDocsDiv);
            return;
        }
        
        // Add each document to the UI
        documents.forEach(doc => {
            const docDiv = document.createElement('div');
            docDiv.className = 'context-item';
            
            const scoreDiv = document.createElement('div');
            scoreDiv.className = 'score';
            scoreDiv.textContent = `Relevance: ${(doc.score * 100).toFixed(1)}%`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'content';
            contentDiv.textContent = doc.content || 'No content available';
            
            docDiv.appendChild(scoreDiv);
            docDiv.appendChild(contentDiv);
            
            contextContainer.appendChild(docDiv);
        });
    }

    async function updateSessionsList() {
        // Reload all sessions to refresh the list
        await loadSessions();
        
        // The highlighting is now done by highlightCurrentSession
    }
    
    function highlightCurrentSession() {
        // Highlight the current session without reloading all sessions
        if (currentSessionId) {
            const sessionElements = document.querySelectorAll('.session-item');
            sessionElements.forEach(el => {
                if (el.dataset.sessionId === currentSessionId) {
                    el.classList.add('active');
                } else {
                    el.classList.remove('active');
                }
            });
            localStorage.setItem('currentSessionId', currentSessionId);
        }
    }
    
    function addSessionToList(session) {
        const sessionItem = document.createElement('div');
        sessionItem.className = 'session-item';
        sessionItem.dataset.sessionId = session.id;
        
        // Format the date for display
        const date = new Date(session.created_at);
        const formattedDate = date.toLocaleString();
        
        sessionItem.innerHTML = `
            <div class="session-info">
                <div class="session-date">${formattedDate}</div>
                <div class="session-id">${session.id.substring(0, 8)}...</div>
            </div>
            <div class="session-actions">
                <button class="delete-session-btn" title="Delete session">×</button>
            </div>
        `;
        
        // Add active class if this is the current session
        if (session.id === currentSessionId) {
            sessionItem.classList.add('active');
        }
        
        // Add click event to load this session
        sessionItem.addEventListener('click', async (e) => {
            // Don't trigger if clicking the delete button
            if (e.target.classList.contains('delete-session-btn')) {
                return;
            }
            
            currentSessionId = session.id;
            await loadSessionHistory(currentSessionId);
        });
        
        // Add delete button functionality
        const deleteBtn = sessionItem.querySelector('.delete-session-btn');
        deleteBtn.addEventListener('click', async (e) => {
            e.stopPropagation(); // Prevent session selection
            
            if (confirm('Are you sure you want to delete this session?')) {
                try {
                    const response = await fetch(`/api/v1/session/${session.id}`, {
                        method: 'DELETE'
                    });
                    
                    if (response.ok) {
                        // Remove from UI
                        sessionItem.remove();
                        
                        // If this was the current session, load another one
                        if (currentSessionId === session.id) {
                            currentSessionId = null;
                            localStorage.removeItem('currentSessionId');
                            
                            // Load the first available session or create a new one
                            await loadSessions();
                        }
                    } else {
                        alert('Failed to delete session');
                    }
                } catch (error) {
                    console.error('Error deleting session:', error);
                    alert('Error deleting session');
                }
            }
        });
        
        sessionsContainer.appendChild(sessionItem);
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function showLoading(show) {
        if (show) {
            loadingIndicator.classList.add('active');
        } else {
            loadingIndicator.classList.remove('active');
        }
    }
});