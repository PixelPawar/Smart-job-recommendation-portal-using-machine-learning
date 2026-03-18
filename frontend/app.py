import streamlit as st
import streamlit.components.v1 as components
import requests

# Configure page settings
st.set_page_config(
    page_title="Smart Job Recommendation Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Interactive 3D Scene (Three.js with drag support) ----
interactive_3d = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin:0; padding:0; }
  body { background: #0e1117; overflow:hidden; cursor:grab; }
  body.dragging { cursor:grabbing; }
  canvas { display:block; }
</style>
</head>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/400, 0.1, 100);
camera.position.z = 10;

const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:false });
renderer.setSize(window.innerWidth, 400);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0e1117, 1);
document.body.appendChild(renderer.domElement);

// Materials
const cyan = new THREE.MeshPhongMaterial({ color:0x00E5FF, wireframe:true, transparent:true, opacity:0.5 });
const blue = new THREE.MeshPhongMaterial({ color:0x007BFF, wireframe:true, transparent:true, opacity:0.4 });
const green = new THREE.MeshPhongMaterial({ color:0x00FFAA, wireframe:true, transparent:true, opacity:0.4 });
const teal = new THREE.MeshPhongMaterial({ color:0x00E5FF, wireframe:true, transparent:true, opacity:0.3 });

// Lights
const ambient = new THREE.AmbientLight(0xffffff, 0.3);
scene.add(ambient);
const point = new THREE.PointLight(0x00E5FF, 1, 50);
point.position.set(5, 5, 5);
scene.add(point);

// Objects
const cube = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), cyan);
cube.position.set(-5, 1.5, 0);
cube.userData = { baseY: 1.5, floatOffset: 0 };
scene.add(cube);

const sphere = new THREE.Mesh(new THREE.SphereGeometry(1.3, 20, 20), blue);
sphere.position.set(5, -1, -1);
sphere.userData = { baseY: -1, floatOffset: 1.5 };
scene.add(sphere);

const prism = new THREE.Mesh(new THREE.ConeGeometry(1.3, 2.5, 3), green);
prism.position.set(2, 2, -2);
prism.userData = { baseY: 2, floatOffset: 3 };
scene.add(prism);

const torus = new THREE.Mesh(new THREE.TorusGeometry(1, 0.3, 12, 32), teal);
torus.position.set(-3, -1.5, -1);
torus.userData = { baseY: -1.5, floatOffset: 2 };
scene.add(torus);

const miniCube = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.8), cyan.clone());
miniCube.material.opacity = 0.3;
miniCube.position.set(0, 0, -3);
miniCube.userData = { baseY: 0, floatOffset: 4 };
scene.add(miniCube);

const octa = new THREE.Mesh(new THREE.OctahedronGeometry(0.9), blue.clone());
octa.material.opacity = 0.35;
octa.position.set(-1, 2.5, -2);
octa.userData = { baseY: 2.5, floatOffset: 5 };
scene.add(octa);

const draggable = [cube, sphere, prism, torus, miniCube, octa];

// Raycaster for drag interaction
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let dragTarget = null;
let isDragging = false;
const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const intersection = new THREE.Vector3();
const offset = new THREE.Vector3();

function getMousePos(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
}

renderer.domElement.addEventListener('mousedown', (e) => {
    getMousePos(e);
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(draggable);
    if (hits.length > 0) {
        dragTarget = hits[0].object;
        isDragging = true;
        document.body.classList.add('dragging');
        plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()).negate(), dragTarget.position);
        raycaster.ray.intersectPlane(plane, intersection);
        offset.copy(dragTarget.position).sub(intersection);
    }
});

renderer.domElement.addEventListener('mousemove', (e) => {
    getMousePos(e);
    if (isDragging && dragTarget) {
        raycaster.setFromCamera(mouse, camera);
        raycaster.ray.intersectPlane(plane, intersection);
        dragTarget.position.copy(intersection.add(offset));
        dragTarget.userData.baseY = dragTarget.position.y;
    }
    // Subtle scene tilt based on mouse
    scene.rotation.y += (mouse.x * 0.15 - scene.rotation.y) * 0.05;
    scene.rotation.x += (mouse.y * 0.08 - scene.rotation.x) * 0.05;
});

renderer.domElement.addEventListener('mouseup', () => {
    isDragging = false;
    dragTarget = null;
    document.body.classList.remove('dragging');
});

renderer.domElement.addEventListener('mouseleave', () => {
    isDragging = false;
    dragTarget = null;
    document.body.classList.remove('dragging');
});

// Animation
function animate() {
    requestAnimationFrame(animate);
    const t = Date.now() * 0.001;
    draggable.forEach((obj, i) => {
        obj.rotation.x += 0.003 + i * 0.001;
        obj.rotation.y += 0.005 + i * 0.0008;
        if (!isDragging || dragTarget !== obj) {
            obj.position.y = obj.userData.baseY + Math.sin(t + obj.userData.floatOffset) * 0.3;
        }
    });
    // Move light to follow mouse for dynamic lighting
    point.position.x = mouse.x * 8;
    point.position.y = mouse.y * 5;
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / 400;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, 400);
});
</script>
</body>
</html>
"""
components.html(interactive_3d, height=400)

# Custom CSS for the main Streamlit UI
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

        .main .block-container {
            position: relative;
            z-index: 1;
        }
        .main {
            background-color: #0e1117;
            font-family: 'Inter', sans-serif;
        }
        h1 {
            color: #ffffff;
            font-weight: 800;
            text-align: center;
            margin-bottom: 10px;
            background: -webkit-linear-gradient(45deg, #00E5FF, #007BFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* ---- Input Box Glassmorphism ---- */
        .stTextInput > div > div > input {
            border-radius: 14px;
            border: 1px solid rgba(0, 229, 255, 0.15);
            background: linear-gradient(135deg, rgba(20, 25, 40, 0.6), rgba(10, 10, 15, 0.8));
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            color: #ffffff;
            padding: 14px 18px;
            font-size: 15px;
            font-family: 'Inter', sans-serif;
            transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.2s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }
        .stTextInput > div > div > input:focus {
            border-color: #00E5FF;
            box-shadow: 0 0 18px rgba(0, 229, 255, 0.3), inset 0 0 8px rgba(0, 229, 255, 0.05);
            transform: translateY(-1px);
        }
        .stTextInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.35);
        }
        .stTextInput label {
            color: #a0a0a0 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            letter-spacing: 0.5px;
        }

        /* ---- Center-Aligned Glowing Button ---- */
        /* Target ALL parent wrappers Streamlit puts around buttons */
        [data-testid="stButton"],
        div.stButton,
        div.row-widget.stButton {
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
            margin: 0 auto !important;
            text-align: center !important;
        }
        /* Force the element-container holding the button to be full-width */
        [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
        }
        /* Also center the element-container that wraps the button */
        div[data-testid="stButton"] {
            display: flex !important;
            justify-content: center !important;
        }
        div.stButton > button,
        [data-testid="stButton"] > button,
        [data-testid="baseButton-secondary"] {
            width: 320px !important;
            border-radius: 14px !important;
            height: 54px !important;
            font-size: 17px !important;
            font-weight: 700 !important;
            font-family: 'Inter', sans-serif !important;
            letter-spacing: 0.5px !important;
            background: linear-gradient(135deg, #007BFF, #00E5FF) !important;
            color: white !important;
            border: 1px solid rgba(0, 229, 255, 0.3) !important;
            box-shadow: 0 8px 25px rgba(0, 123, 255, 0.3), 0 0 15px rgba(0, 229, 255, 0.1) !important;
            transition: transform 0.2s ease, box-shadow 0.3s ease !important;
            cursor: pointer !important;
            margin: 0 auto !important;
        }
        div.stButton > button:hover,
        [data-testid="baseButton-secondary"]:hover {
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 0 15px 40px rgba(0, 123, 255, 0.5), 0 0 30px rgba(0, 229, 255, 0.3) !important;
            border-color: #00E5FF !important;
            color: white !important;
        }
        div.stButton > button:active,
        [data-testid="baseButton-secondary"]:active {
            transform: translateY(0px) scale(0.98) !important;
            box-shadow: 0 5px 15px rgba(0, 123, 255, 0.3) !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Smart Job Recommendation Portal")

st.markdown("<p style='text-align: center; color: #a0a0a0; margin-bottom: 40px;'>Find your dream job powered by Machine Learning.</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    skills = st.text_input("Skills", placeholder="e.g. Python, Machine Learning")
with col2:
    location = st.text_input("Location (optional)", placeholder="e.g. US, UK")
with col3:
    experience = st.text_input("Experience (years)", placeholder="e.g. 0, 1, 3, 5")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Get Recommendations"):
    if not skills.strip():
        st.error("Please enter skills to get recommendations.")
    else:
        with st.spinner("Analyzing matches..."):
            try:
                # Make POST request to backend API
                response = requests.post("http://127.0.0.1:5000/recommend", json={
                    "skills": skills,
                    "location": location,
                    "experience": experience
                })
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "error" in data:
                        st.error(data["error"])
                    elif len(data) == 0:
                        st.warning("No matching jobs found. Try adjusting your skills or experience.")
                    else:
                        st.success(f"Found {len(data)} job recommendations!")
                        
                        # Generate HTML for 3D Cards
                        cards_html = """
                        <!DOCTYPE html>
                        <html>
                        <head>
                        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
                        <style>
                        body {
                            margin: 0;
                            padding: 20px 10px;
                            font-family: 'Inter', sans-serif;
                            display: flex;
                            flex-wrap: wrap;
                            justify-content: center;
                            gap: 30px;
                            background-color: #0e1117;
                        }
                        .job-card-container {
                            perspective: 1200px;
                            width: calc(33.333% - 40px);
                            min-width: 280px;
                            max-width: 350px;
                        }
                        .job-card {
                            background: linear-gradient(135deg, rgba(20, 25, 40, 0.6), rgba(10, 10, 15, 0.8));
                            backdrop-filter: blur(15px);
                            -webkit-backdrop-filter: blur(15px);
                            padding: 30px;
                            border-radius: 20px;
                            border: 1px solid rgba(255, 255, 255, 0.08);
                            border-top: 1px solid rgba(0, 229, 255, 0.3);
                            border-left: 1px solid rgba(0, 229, 255, 0.2);
                            box-shadow: 0 15px 35px rgba(0,0,0,0.6), inset 0 0 20px rgba(0, 229, 255, 0.05);
                            transform-style: preserve-3d;
                            transition: transform 0.1s ease-out, box-shadow 0.3s ease, border-color 0.3s ease;
                            will-change: transform;
                            position: relative;
                            color: white;
                            height: 100%;
                            box-sizing: border-box;
                        }
                        .job-card::before {
                            content: '';
                            position: absolute;
                            inset: 0;
                            border-radius: 20px;
                            padding: 2px;
                            background: linear-gradient(135deg, rgba(0,229,255,0.5), transparent 50%, rgba(0,123,255,0.5));
                            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
                            -webkit-mask-composite: xor;
                            mask-composite: exclude;
                            opacity: 0;
                            transition: opacity 0.3s ease;
                            pointer-events: none;
                        }
                        .job-card::after {
                            content: '';
                            position: absolute;
                            top: 0; right: 0; bottom: 0; left: 0;
                            border-radius: 20px;
                            background: radial-gradient(600px circle at var(--mouse-x) var(--mouse-y), rgba(0,229,255,0.15), transparent 40%);
                            z-index: 1;
                            pointer-events: none;
                            mix-blend-mode: color-dodge;
                        }
                        .job-card:hover {
                            box-shadow: 0 30px 60px rgba(0,0,0,0.8), 0 0 40px rgba(0, 229, 255, 0.3);
                        }
                        .job-card:hover::before {
                            opacity: 1;
                        }
                        .job-card-content {
                            transform-style: preserve-3d;
                            position: relative;
                            z-index: 2;
                        }
                        .job-card h3 {
                            margin-top: 0;
                            margin-bottom: 15px;
                            color: #ffffff;
                            font-size: 22px;
                            font-weight: 800;
                            transform: translateZ(50px);
                            text-shadow: 0 5px 15px rgba(0,0,0,0.5);
                        }
                        .job-card .badge {
                            display: inline-block;
                            padding: 5px 10px;
                            border-radius: 20px;
                            background: rgba(0, 229, 255, 0.1);
                            color: #00E5FF;
                            font-size: 12px;
                            font-weight: 600;
                            margin-bottom: 15px;
                            transform: translateZ(40px);
                        }
                        .detail-row {
                            display: flex;
                            align-items: center;
                            margin: 10px 0;
                            transform: translateZ(30px);
                        }
                        .detail-label {
                            color: #888;
                            font-size: 14px;
                            width: 80px;
                            font-weight: 600;
                        }
                        .detail-value {
                            color: #ddd;
                            font-size: 15px;
                            font-weight: 400;
                        }
                        </style>
                        </head>
                        <body>
                        """
                        
                        for job in data:
                            title = job.get("title", "No Title")
                            location = job.get("location", "N/A")
                            industry = job.get("industry", "N/A")
                            exp = job.get("required_experience", "N/A")
                            
                            cards_html += f"""
                            <div class="job-card-container">
                                <div class="job-card" onmousemove="handleTilt(event, this)" onmouseleave="resetTilt(this)">
                                    <div class="job-card-content">
                                        <div class="badge">{industry}</div>
                                        <h3>{title}</h3>
                                        <div class="detail-row">
                                            <span class="detail-label">Location</span>
                                            <span class="detail-value">{location}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span class="detail-label">Experience</span>
                                            <span class="detail-value">{exp}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """
                            
                        cards_html += """
                        <script>
                        function handleTilt(e, card) {
                            const rect = card.getBoundingClientRect();
                            const x = e.clientX - rect.left;
                            const y = e.clientY - rect.top;
                            
                            // Mouse position for radial gradient highlight
                            card.style.setProperty('--mouse-x', `${x}px`);
                            card.style.setProperty('--mouse-y', `${y}px`);
                            
                            // 3D Tilt calculation
                            const centerX = rect.width / 2;
                            const centerY = rect.height / 2;
                            
                            const rotateX = ((y - centerY) / centerY) * -15;
                            const rotateY = ((x - centerX) / centerX) * 15;
                            
                            card.style.transform = `perspective(1200px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.05, 1.05, 1.05)`;
                        }
                        function resetTilt(card) {
                            card.style.transform = "perspective(1200px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)";
                        }
                        </script>
                        </body>
                        </html>
                        """
                        
                        # Calculate appropriate iframe height based on number of cards
                        # Assuming 3 cards max per row on wide screens
                        rows = (len(data) + 2) // 3
                        # Each card is roughly 250px tall, plus gap and padding
                        iframe_height = max(350, rows * 320)
                        
                        # Render the custom HTML components in to the UI
                        components.html(cards_html, height=iframe_height, scrolling=True)
                        
                else:
                    st.error(f"Error connecting to model server: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend server. Is it running on http://127.0.0.1:5000?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
