import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ==============================================================================
# PAGE CONFIGURATION & THEME SETUP
# Desktop Widescreen PC Viewport
# ==============================================================================
st.set_page_config(
    page_title="CNS Healthcare Psychological Services Appraisal Web App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate-Executive Styling
st.markdown("""
<style>
    /* Main Canvas Background */
    .stApp {
        background-color: #0F172A !important; /* Deep Slate Navy */
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%);
        border: 1px solid #3B82F6;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        color: #FFFFFF !important;
        font-size: 24px !important;
        font-weight: 800 !important;
        margin-bottom: 4px !important;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: #93C5FD !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    
    /* Audit Card Styling */
    .audit-card {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        transition: border-color 0.2s ease;
    }
    
    .audit-card:hover {
        border-color: #3B82F6 !important;
    }
    
    /* Badges */
    .badge-policy {
        background-color: #1E3A8A;
        color: #DBEAFE;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 8px;
        border: 1px solid #2563EB;
        font-family: monospace;
    }
    
    .badge-target {
        background-color: #065F46;
        color: #D1FAE5;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 8px;
        margin-left: 6px;
        border: 1px solid #059669;
        font-family: monospace;
    }

    /* Remediation Alerts */
    .remedy-box {
        background-color: rgba(217, 119, 6, 0.15) !important;
        border-left: 4px solid #F59E0B !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin-top: 10px !important;
        font-size: 12px !important;
        color: #FDE68A !important;
        line-height: 1.4;
    }
    
    .cascade-alert {
        background-color: rgba(220, 38, 38, 0.15) !important;
        border-left: 5px solid #EF4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        margin-bottom: 16px !important;
        font-size: 12px !important;
        color: #FCA5A5 !important;
    }
    
    /* Streamlit UI Overrides */
    div.stPopover > button {
        background-color: #0F172A !important;
        border: 1px solid #3B82F6 !important;
        color: #60A5FA !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    
    div.stPopover > button:hover {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CACHED DATA SOURCES & REGULATORY MAPPINGS
# ==============================================================================
@st.cache_data
def load_claims_telemetry():
    """Simulates/caches 12-month query of 1,420 historical psychological testing claims."""
    denial_data = pd.DataFrame({
        "Denial Reason / Vulnerability": [
            "Same-Day Provider/Tech NCCI Error (96136/96138)",
            "Telehealth POS/Modifier Mismatch (POS 02/10, Mod 95)",
            "Exceeded Prior Authorization Limit (Meridian 8-Hr Cap)",
            "Missing CPT 96130 Interactive Feedback Note",
            "Unattached Diagnostic Justification / BTP Clearance"
        ],
        "Claim Count": [384, 298, 245, 182, 111],
        "Financial Exposure ($)": [76800, 53640, 49000, 36400, 22200],
        "Root Cause": [
            "Missing Modifier XE/59 & Medical Necessity Addendum",
            "EHR missing auto-append rules for telehealth feedback",
            "Lack of hard-stop scheduling lock at 8-hour threshold",
            "Provider locked note prior to feedback completion",
            "Lack of integrated physical exam / FBA attachment"
        ]
    })
    
    tat_data = pd.DataFrame({
        "Clinic Site": ["Detroit (Wayne)", "Pontiac (Oakland)", "Southfield (Oakland)", "Novi (Oakland)", "Eastpointe (Macomb)"],
        "Referral-to-Auth (Days)": [3.2, 2.8, 2.5, 2.1, 3.0],
        "Auth-to-Testing (Days)": [8.5, 9.1, 7.2, 6.8, 8.9],
        "Testing-to-Signed Report (Days)": [14.2, 12.8, 11.5, 10.2, 13.5],
        "Total Turnaround Time (Days)": [25.9, 24.7, 21.2, 19.1, 25.4]
    })
    return denial_data, tat_data

@st.cache_resource
def load_regulatory_framework():
    """Caches authoritative policy frameworks and task definitions."""
    return {
        "Phase 1": {
            "title": "Phase 1: Discovery, Baseline & Pipeline Mapping (Days 1–30)",
            "objective": "Verify 100% LARA LLP supervision logs, map referral-to-authorization pipelines, and audit CPT 96130 feedback integrity.",
            "strengths": "**Learner® & Intellection®**: Absorb dense LARA statutes (MCL 333.18223) and Board of Psychology rules; conduct deep root-cause analysis on documentation gaps without premature action bias.",
            "tasks": {
                "p1_t1": {
                    "label": "Verify LLP Supervision Logs & Form LARA/BPL Rev. 6/25",
                    "policy": "MCL 333.18223 & LARA Rule 338.2569",
                    "target": "100% compliant logs; 4 hrs/mo LP supervision",
                    "desc": "Audit 100% of supervisory files for LLPs/TLLPs across all clinics. Ensure 4 hours/month face-to-face LP supervision is logged on Form LARA/BPL Rev. 6/25 and LP co-signatures are present in EHR.",
                    "rationale": "Unsupervised LLP clinical activity is a direct violation of licensure law, exposing CNS Healthcare to state disciplinary actions and retroactive Medicaid payment recoupments.",
                    "remediation": "Immediately halt unsupervised LLP billing. Centralize supervision logs in an automated HR tracking portal and enforce EHR co-signature locks before claim release."
                },
                "p1_t2": {
                    "label": "Audit CPT 96130 Documentation Integrity (Interactive Feedback)",
                    "policy": "CPT 96130 Guidelines & Medicare LCD",
                    "target": "100% presence of feedback note in sample",
                    "desc": "Extract and audit 30 completed psychological/neuropsychological evaluation charts across pediatric, adult, and geriatric caseloads.",
                    "rationale": "Billing evaluation code CPT 96130 without documenting face-to-face interactive feedback constitutes billing non-compliance and FWA vulnerability.",
                    "remediation": "Disclose deficient billing encounters. Deploy mandatory NextGen EHR templates requiring a timestamped 'Interactive Feedback and Clinical Decision Making' section prior to note locking."
                },
                "p1_t3": {
                    "label": "Map Referral-to-Authorization Pipeline & PIHP Portals",
                    "policy": "SAMHSA CCBHC Criteria & MDHHS Handbook",
                    "target": "Map 100% of pipeline lifecycle & velocity",
                    "desc": "Shadow intake staff tracking testing referrals from request through EHR queues to PIHP authorization portals (MHWIN/CHAMPS).",
                    "rationale": "Extended waiting periods delay diagnostic testing, violating CCBHC timely access benchmarks and triggering state Corrective Action Plans.",
                    "remediation": "Map administrative handoffs, resolve bottlenecks in PIHP portals, and establish open-access triage scheduling blocks."
                },
                "p1_t4": {
                    "label": "Audit Access-to-Care Timeliness Benchmarks",
                    "policy": "MDHHS CCBHC Timely Access Mandate",
                    "target": "Crisis <3h, Urgent 1d, Routine 14d, Comp Eval 60d",
                    "desc": "Recalibrate clinic scheduling queues to strictly enforce state access benchmarks: Crisis (<3 hours), Urgent (1 business day), Routine (14 calendar days), Comprehensive Eval (60 days).",
                    "rationale": "Exceeding access benchmarks violates CCBHC demonstration standards and threatens prospective payment system (PPS-1) certification.",
                    "remediation": "Deploy Stepped-Care triage screenings (CPT 96127) at intake to manage waitlists and provide active interim care coordination."
                }
            }
        },
        "Phase 2": {
            "title": "Phase 2: Operational Analytics, Financial Audit & Gap Assessment (Days 31–60)",
            "objective": "Analyze CPT 96130–96139 denial codes, calculate median Turnaround Times against 1/14/60-day MDHHS mandates, and conduct an Overhead vs. PPS ROI analysis.",
            "strengths": "**Ideation® & Individualization®**: Design custom voice macros in Dragon Medical One and NextGen EHR templates; tailor clinical coaching to each clinician's unique writing style.",
            "tasks": {
                "p2_t1": {
                    "label": "Audit 12-Month CPT 96130–96139 Remittance Denial Data",
                    "policy": "RCM 835 Remittance Guidelines",
                    "target": "Identify top 3 testing denial reason codes",
                    "desc": "Extract and analyze a 12-month claims dataset to isolate top clearinghouse edit rejections (CO-97 bundled codes, CO-50 medical necessity).",
                    "rationale": "Unresolved clearinghouse denials cause massive revenue leakage and administrative burden for retroactive manual billing appeals.",
                    "remediation": "Correct structural billing errors at point of scheduling and update front-end EHR clearinghouse validation rules."
                },
                "p2_t2": {
                    "label": "Execute NCCI Modifier XE/59 Audit (Same-Day Billing)",
                    "policy": "CMS National Correct Coding Initiative (NCCI)",
                    "target": "100% same-day provider/tech modifier accuracy",
                    "desc": "Audit same-day psychologist administration (96136) and technician administration (96138) claims for proper Modifier XE/59 application.",
                    "rationale": "Same-day testing administration without a modifier triggers automated NCCI denials and FWA compliance scrutiny.",
                    "remediation": "Hardcode NCCI validation rules in EHR billing modules to flag same-day testing code pairs and require a Medical Necessity Addendum."
                },
                "p2_t3": {
                    "label": "Audit Telehealth Modifier & Place of Service Compliance",
                    "policy": "MDHHS Telehealth Guidelines",
                    "target": "100% virtual feedback modifier compliance",
                    "desc": "Verify virtual feedback claims append Modifier 95/GT and POS 02/10, while enforcing hard-stops against remote testing administration.",
                    "rationale": "Missing telehealth modifiers or improper POS codes cause immediate claim rejections and suppress realization rates.",
                    "remediation": "Configure EHR telehealth modules to auto-append POS 02/10 and Modifier 95 when virtual links are generated, and block remote admin for unvalidated tests."
                },
                "p2_t4": {
                    "label": "Evaluate PPS Encounter Splitting & Multi-Day Testing Rules",
                    "policy": "CMS & MDHHS FWA Unbundling Guidelines",
                    "target": "100% documented clinical justification for multi-day testing",
                    "desc": "Audit multi-day testing sessions to ensure scheduling is strictly driven by clinical necessity (patient fatigue, pediatric ADHD, motor limits).",
                    "rationale": "Splitting testing across multiple days solely to generate extra daily PPS encounter claims (`T1040`) violates CMS/MDHHS rules and invites recoupments.",
                    "remediation": "Enforce a mandatory EHR selection titled 'Justification for Multi-Day Testing' prior to scheduling follow-up evaluation sessions."
                },
                "p2_t5": {
                    "label": "Calculate Clinician Report Turnaround Times (TAT)",
                    "policy": "CARF Quality Timeliness Criteria",
                    "target": "Median TAT < 14 days; total eval < 60 days",
                    "desc": "Extract EHR timestamps to measure mean/median days from test completion to final signed report, segmenting clinician performance.",
                    "rationale": "Extended report TATs delay psychiatric prescriptions and therapy entry, violating CCBHC care coordination mandates.",
                    "remediation": "Segment EHR timestamps into 3 intervals (ref-to-auth, auth-to-test, test-to-signed) and deliver targeted coaching to outlying write-times."
                },
                "p2_t6": {
                    "label": "Conduct Testing Kit Overhead vs. PPS-1 ROI Analysis",
                    "policy": "CCBHC PPS Cost Allocation Guidelines",
                    "target": "Establish cost-per-assessment ratio",
                    "desc": "Cross-reference vendor invoices for paper kits and digital scoring licenses (Pearson Q-interactive, PARiConnect) against daily PPS revenues (`T1040`).",
                    "rationale": "Unmonitored diagnostic kit and licensing expenses create unrecognized department deficits under flat daily PPS encounter rates.",
                    "remediation": "Transition completely to digital scoring platforms to lower material overhead and reduce administration time by up to 40%."
                }
            }
        },
        "Phase 3": {
            "title": "Phase 3: Strategic Synthesis, CQI Framework & Executive Roadmap (Days 61–90)",
            "objective": "Implement Stepped-Care Assessment Protocol, launch automated EHR KPI dashboard, embed testing results into PCPs, and deliver 12-month roadmap.",
            "strengths": "**Strategic®**: Synthesize findings across all five audit pillars into a razor-sharp business case for executive leadership, prioritizing top capital investments.",
            "tasks": {
                "p3_t1": {
                    "label": "Implement Stepped-Care Assessment Triage Protocol",
                    "policy": "SAMHSA CCBHC Core Service #2",
                    "target": "100% referrals triaged via brief screening first",
                    "desc": "Deploy a clinical algorithm filtering low-acuity cases via brief screenings (CPT 96127) at intake, reserving multi-hour batteries for complex SMI/SED diagnosis.",
                    "rationale": "Conducting multi-day diagnostic testing for low-acuity referrals wastes psychologist FTE capacity, inflating waitlists for high-acuity consumers.",
                    "remediation": "Develop and approve the Stepped-Care Assessment clinical algorithm, establishing a strict diagnostic intake screen to preserve testing resources."
                },
                "p3_t2": {
                    "label": "Configure Live EHR Assessment KPI Dashboard",
                    "policy": "CCBHC Continuous Quality Improvement (CQI) Plan",
                    "target": "Live tracking of 5 core clinical-financial KPIs",
                    "desc": "Build an EHR-integrated Business Intelligence dashboard tracking weekly referral volume, waitlist duration, median TAT, denial rates, and cost-per-assessment.",
                    "rationale": "Lack of real-time visual tracking leads to unrecognized bottlenecks and billing errors, resulting in compounding financial loss.",
                    "remediation": "Coordinate with IT to configure a live BI dashboard giving leadership immediate visibility over clinician performance and claim denials."
                },
                "p3_t3": {
                    "label": "Embed Diagnostic Results into Person-Centered Plans (PCP/IPOS)",
                    "policy": "MDHHS & SAMHSA Care Criteria",
                    "target": "100% integration of testing recommendations in IPOS",
                    "desc": "Establish automated EHR workflows ensuring testing formulations directly populate the consumer's Individualized Plan of Service (IPOS).",
                    "rationale": "If diagnostic reports fail to influence the PCP, testing operates as an isolated, high-cost administrative exercise.",
                    "remediation": "Implement automated notifications from the psychology EHR queue flagging recommendations directly to the assigned case manager."
                },
                "p3_t4": {
                    "label": "Deliver Executive Appraisal Report & 12-Month Strategic Roadmap",
                    "policy": "CCBHC Program Requirement #6",
                    "target": "Formal Board submission & budget approval",
                    "desc": "Synthesize all audit findings into the formal 'State of Psychological Testing' report and present the 12-month strategic optimization roadmap.",
                    "rationale": "Failure to plan for long-term capital investments results in operational stagnation, clinician burnout, and unresolved financial leaks.",
                    "remediation": "Present the 12-month roadmap to executive leadership to secure budget allocation and strategic alignment for top operational priorities."
                }
            }
        }
    }

# Initialize Session State Database
if "appraisal_state" not in st.session_state:
    st.session_state.appraisal_state = {}
    reg_db = load_regulatory_framework()
    for phase_key, phase_val in reg_db.items():
        for task_key in phase_val["tasks"].keys():
            st.session_state.appraisal_state[task_key] = {
                "status": "Pending",
                "notes": ""
            }

denial_df, tat_df = load_claims_telemetry()
reg_db = load_regulatory_framework()

# ==============================================================================
# HEADER BANNER & APP NAVIGATION
# ==============================================================================
st.markdown("""
<div class="header-banner">
    <div class="header-title">CNS Healthcare Psychological Services Appraisal Portal</div>
    <div class="header-subtitle">Executive Command Terminal • 90-Day Execution Plan & Compliance Audit Engine</div>
</div>
""", unsafe_allow_html=True)

nav_tabs = st.tabs([
    " Phase 1: Discovery",
    " Phase 2: Analytics",
    " Phase 3: Strategic Roadmap",
    " Executive Memorandum & Risk Matrix",
    " Supervisor Oversight & Telemetry"
])

# ==============================================================================
# TAB 1: PHASE 1 DISCOVERY
# ==============================================================================
with nav_tabs[0]:
    st.markdown(f"### **{reg_db['Phase 1']['title']}**")
    st.info(f"**Objective**: {reg_db['Phase 1']['objective']}")
    st.markdown(f"**CliftonStrengths Alignment**: {reg_db['Phase 1']['strengths']}")
    
    with st.popover(" VIEW PHASE 1 DIAGNOSTIC BLUEPRINT"):
        st.markdown("**Phase 1 Strategic Scope**")
        st.write("Focuses on regulatory discovery and clinical baseline mapping across Wayne, Oakland, and Macomb clinics. Under CCBHC guidelines, establishing an empirical baseline is critical prior to system re-engineering.")

    st.markdown("---")
    
    for t_key, t_val in reg_db["Phase 1"]["tasks"].items():
        st.markdown('<div class="audit-card">', unsafe_allow_html=True)
        col_a, col_b = st.columns([3, 2])
        
        with col_a:
            st.markdown(f'<span class="badge-policy">{t_val["policy"]}</span> <span class="badge-target">KPI: {t_val["target"]}</span>', unsafe_allow_html=True)
            st.markdown(f"#### **{t_val['label']}**")
            st.write(t_val["desc"])
            
        with col_b:
            with st.popover(" TASK DIAGNOSTICS"):
                st.markdown(f"**Policy Source**: {t_val['policy']}")
                st.markdown(f"**Appraisal Rationale**: {t_val['rationale']}")
            
            curr_status = st.session_state.appraisal_state[t_key]["status"]
            status_opts = ["Pending", "Compliant", "Outside of Compliance"]
            
            new_status = st.selectbox(
                "Diagnostic Status",
                status_opts,
                index=status_opts.index(curr_status),
                key=f"sel_{t_key}"
            )
            st.session_state.appraisal_state[t_key]["status"] = new_status
            
            user_notes = st.text_input("Audit Notes / Findings", value=st.session_state.appraisal_state[t_key]["notes"], key=f"note_{t_key}")
            st.session_state.appraisal_state[t_key]["notes"] = user_notes
            
        if new_status == "Outside of Compliance":
            st.markdown(f'<div class="remedy-box">⚠️ <strong>REMEDIATION DIRECTIVE:</strong> {t_val["remediation"]}</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2: PHASE 2 ANALYTICS
# ==============================================================================
with nav_tabs[1]:
    st.markdown(f"### **{reg_db['Phase 2']['title']}**")
    st.info(f"**Objective**: {reg_db['Phase 2']['objective']}")
    st.markdown(f"**CliftonStrengths Alignment**: {reg_db['Phase 2']['strengths']}")
    
    # Cascade check from Phase 1
    p1_gaps = [tk for tk in reg_db["Phase 1"]["tasks"].keys() if st.session_state.appraisal_state[tk]["status"] == "Outside of Compliance"]
    if p1_gaps:
        st.markdown('<div class="cascade-alert">🚨 <strong>CRITICAL CARRYOVER ALERT: UNRESOLVED PHASE 1 RISKS DETECTED!</strong><br>The following baseline components remain out of compliance, threatening Phase 2 operational validity: ' + ", ".join([reg_db["Phase 1"]["tasks"][tk]["label"] for tk in p1_gaps]) + '</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    for t_key, t_val in reg_db["Phase 2"]["tasks"].items():
        st.markdown('<div class="audit-card">', unsafe_allow_html=True)
        col_a, col_b = st.columns([3, 2])
        
        with col_a:
            st.markdown(f'<span class="badge-policy">{t_val["policy"]}</span> <span class="badge-target">KPI: {t_val["target"]}</span>', unsafe_allow_html=True)
            st.markdown(f"#### **{t_val['label']}**")
            st.write(t_val["desc"])
            
        with col_b:
            with st.popover(" TASK DIAGNOSTICS"):
                st.markdown(f"**Policy Source**: {t_val['policy']}")
                st.markdown(f"**Appraisal Rationale**: {t_val['rationale']}")
            
            curr_status = st.session_state.appraisal_state[t_key]["status"]
            status_opts = ["Pending", "Compliant", "Outside of Compliance"]
            
            new_status = st.selectbox(
                "Diagnostic Status",
                status_opts,
                index=status_opts.index(curr_status),
                key=f"sel_{t_key}"
            )
            st.session_state.appraisal_state[t_key]["status"] = new_status
            
            user_notes = st.text_input("Audit Notes / Findings", value=st.session_state.appraisal_state[t_key]["notes"], key=f"note_{t_key}")
            st.session_state.appraisal_state[t_key]["notes"] = user_notes
            
        if new_status == "Outside of Compliance":
            st.markdown(f'<div class="remedy-box">⚠️ <strong>REMEDIATION DIRECTIVE:</strong> {t_val["remediation"]}</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 3: PHASE 3 STRATEGIC ROADMAP
# ==============================================================================
with nav_tabs[2]:
    st.markdown(f"### **{reg_db['Phase 3']['title']}**")
    st.info(f"**Objective**: {reg_db['Phase 3']['objective']}")
    st.markdown(f"**CliftonStrengths Alignment**: {reg_db['Phase 3']['strengths']}")
    
    p2_gaps = [tk for tk in reg_db["Phase 2"]["tasks"].keys() if st.session_state.appraisal_state[tk]["status"] == "Outside of Compliance"]
    if p2_gaps:
        st.markdown('<div class="cascade-alert">🚨 <strong>CRITICAL CARRYOVER ALERT: UNRESOLVED PHASE 2 GAPS DETECTED!</strong><br>The following operational gaps remain unresolved, impacting Phase 3 roadmap stability: ' + ", ".join([reg_db["Phase 2"]["tasks"][tk]["label"] for tk in p2_gaps]) + '</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    for t_key, t_val in reg_db["Phase 3"]["tasks"].items():
        st.markdown('<div class="audit-card">', unsafe_allow_html=True)
        col_a, col_b = st.columns([3, 2])
        
        with col_a:
            st.markdown(f'<span class="badge-policy">{t_val["policy"]}</span> <span class="badge-target">KPI: {t_val["target"]}</span>', unsafe_allow_html=True)
            st.markdown(f"#### **{t_val['label']}**")
            st.write(t_val["desc"])
            
        with col_b:
            with st.popover(" TASK DIAGNOSTICS"):
                st.markdown(f"**Policy Source**: {t_val['policy']}")
                st.markdown(f"**Appraisal Rationale**: {t_val['rationale']}")
            
            curr_status = st.session_state.appraisal_state[t_key]["status"]
            status_opts = ["Pending", "Compliant", "Outside of Compliance"]
            
            new_status = st.selectbox(
                "Diagnostic Status",
                status_opts,
                index=status_opts.index(curr_status),
                key=f"sel_{t_key}"
            )
            st.session_state.appraisal_state[t_key]["status"] = new_status
            
            user_notes = st.text_input("Audit Notes / Findings", value=st.session_state.appraisal_state[t_key]["notes"], key=f"note_{t_key}")
            st.session_state.appraisal_state[t_key]["notes"] = user_notes
            
        if new_status == "Outside of Compliance":
            st.markdown(f'<div class="remedy-box">⚠️ <strong>REMEDIATION DIRECTIVE:</strong> {t_val["remediation"]}</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 4: EXECUTIVE MEMORANDUM & RISK MATRIX
# ==============================================================================
with nav_tabs[3]:
    st.markdown("### **Executive Status Appraisal Memorandum**")
    st.caption("Aggregated Compliance Status & Active Remediation Directives for Supervisor Review")
    
    # Calculate counts
    total_tasks = 0
    compliant_items = []
    non_compliant_items = []
    pending_items = []
    
    for phase_key, phase_val in reg_db.items():
        for t_key, t_val in phase_val["tasks"].items():
            total_tasks += 1
            st_val = st.session_state.appraisal_state[t_key]["status"]
            notes_val = st.session_state.appraisal_state[t_key]["notes"]
            item_entry = {"key": t_key, "info": t_val, "phase": phase_val["title"], "notes": notes_val}
            
            if st_val == "Compliant":
                compliant_items.append(item_entry)
            elif st_val == "Outside of Compliance":
                non_compliant_items.append(item_entry)
            else:
                pending_items.append(item_entry)
                
    comp_rate = (len(compliant_items) / total_tasks) * 100 if total_tasks > 0 else 0
    
    # Metrics Scorecard
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Overall Compliance Rate", f"{comp_rate:.1f}%")
    m_col2.metric("Verified Compliant Tasks", len(compliant_items))
    m_col3.metric("Active Non-Compliant Gaps", len(non_compliant_items))
    m_col4.metric("Pending Audits", len(pending_items))
    
    st.progress(comp_rate / 100.0)
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### ** Verified Compliant Operations**")
        if compliant_items:
            for item in compliant_items:
                st.success(f"**{item['info']['label']}**\n\n*Policy*: {item['info']['policy']} | *KPI*: {item['info']['target']}\n\n*Notes*: {item['notes'] if item['notes'] else 'Verified Compliant'}")
        else:
            st.write("No tasks currently marked as compliant.")
            
    with col_right:
        st.markdown("#### ** Active Risk & Mitigation Matrix**")
        if non_compliant_items:
            for item in non_compliant_items:
                st.error(f"**[GAP] {item['info']['label']}**\n\n*Policy*: {item['info']['policy']}\n\n*Systemic Rationale*: {item['info']['rationale']}")
                st.markdown(f'<div class="remedy-box">🛠️ <strong>REMEDIATION DIRECTIVE:</strong> {item["info"]["remediation"]}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.write("No active non-compliant gaps flagged.")

# ==============================================================================
# TAB 5: SUPERVISOR OVERSIGHT & TELEMETRY
# ==============================================================================
with nav_tabs[4]:
    st.markdown("### **Supervisor Oversight & Claims Telemetry**")
    st.caption("Real-Time Analytics, LARA Supervisory Audits & CCBHC Quality Metrics")
    
    # Row 1: Telemetry Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### **12-Month CPT Denial Drivers (1,420 Claims)**")
        fig_donut = px.pie(
            denial_df,
            values="Claim Count",
            names="Denial Reason / Vulnerability",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col_chart2:
        st.markdown("#### **Clinician Report Turnaround Time (TAT) by Clinic**")
        fig_bar = px.bar(
            tat_df,
            x="Clinic Site",
            y=["Referral-to-Auth (Days)", "Auth-to-Testing (Days)", "Testing-to-Signed Report (Days)"],
            title="Segmented Report Write-Time Intervals",
            barmode="stack",
            color_discrete_sequence=["#3B82F6", "#F59E0B", "#10B981"]
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            legend_title_text="Workflow Interval",
            margin=dict(t=30, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    
    # Row 2: LARA Ledger & Quality Measures
    col_lara, col_qbp = st.columns(2)
    
    with col_lara:
        st.markdown("#### **LARA Supervision Ledger & Co-Signature Lock**")
        st.markdown("""
        *   **MCL 333.18223 Mandate**: Minimum 4 hours/month individual face-to-face LP supervision required for all LLPs.
        *   **Form LARA/BPL Rev. 6/25**: Official monthly logs must be signed and archived in central HR.
        *   **EHR Routing Lock**: Claims involving LLP-rendered psychometrics (`96138`) are blocked from release until supervising LP co-signs note.
        """)
