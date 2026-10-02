# AeroVelo Logistics Global — Bill of Materials

**Document ID:** BOM-FLT-H2NEX  
**Revision:** 7.3 (Current)  
**Effective Date:** 2025-08-15  
**Classification:** Internal — Operations, Procurement, and Engineering  
**Product:** H2-Nexus Class Cargo Handling System  
**Applicable Hubs:** Frankfurt (FRA), Dallas-Fort Worth (DFW), Singapore (SIN)

---

## 1. Product Description

The H2-Nexus is a modular cargo handling system designed for high-volume containerized freight operations. It includes automated guided vehicles (AGVs), conveyor systems, and hydraulic lifting mechanisms rated for containers up to 45 feet / 40 tons.

## 2. Bill of Materials — H2-Nexus Standard Configuration

### 2.1 Major Sub-Assemblies

| Sub-Assembly ID | Description | Unit Cost (USD) | Quantity per Unit | Source |
|----------------|-------------|-----------------|-------------------|--------|
| H2-NEX-001 | Hydraulic lifting mechanism (dual-stage) | $185,000 | 4 | Atlas Hydraulic Systems (primary) / Hydratech Industries (alternate) |
| H2-NEX-002 | AGV chassis unit (electric, 12-ton capacity) | $92,000 | 8 | ElectroMove Robotics (primary) / KineticBot Corp (alternate) |
| H2-NEX-003 | Control system integration module | $47,500 | 1 | AVL Software Division (in-house) |
| H2-NEX-004 | Safety sensor array (LiDAR + optical) | $23,000 | 12 | Sentinel Sensors Ltd. (primary) / Optical Guard Inc. (alternate) |
| H2-NEX-005 | Container clamping mechanism (hydraulic) | $34,000 | 6 | Atlas Hydraulic Systems (primary) |
| H2-NEX-006 | Conveyor belt module (modular, 6m) | $18,500 | 10 | BeltFlow Systems (primary) / ConveyTech Ltd. (alternate) |
| H2-NEX-007 | Emergency stop and isolation system | $12,000 | 1 | SafetyFirst Industries (sole source — regulatory requirement) |
| H2-NEX-008 | Power distribution unit (480V 3-phase) | $15,500 | 2 | PowerGrid Solutions (primary) / EuroVolt GmbH (EMEA-only alternate) |

### 2.2 Per-Component Aggregate Costs (Standard Configuration)

| Component | Unit Cost | Quantity | Line Total |
|-----------|-----------|----------|-----------|
| H2-NEX-001 (Hydraulic lifting mechanism) | $185,000 | 4 | **$740,000 per system** |
| H2-NEX-005 (Container clamping mechanism, hydraulic) | $34,000 | 6 | **$204,000 per system** |

Primary source for both H2-NEX-001 and H2-NEX-005 is **Atlas Hydraulic Systems**; Hydratech Industries is the qualified alternate for H2-NEX-001 only.

### 2.3 Total System Cost

| Configuration | Total per Unit (USD) |
|--------------|---------------------|
| Standard (8 AGV, 40ft capacity) | **$1,478,000** |
| Extended (12 AGV, 45ft capacity) | **$1,892,000** |
| Maximum (16 AGV, 45ft + dual clamping) | **$2,340,000** |

## 3. Valve Specifications — H2-Nexus Critical Components

### 3.1 Hydraulic Valve-B (Primary Lifting Circuit)

| Parameter | Specification |
|-----------|--------------|
| Maximum operating pressure | **2,800 psi** (193 bar) |
| Maximum allowable test pressure | 4,200 psi (290 bar) — do NOT exceed |
| Manufacturer | Atlas Hydraulic Systems, Model AHS-4400V |
| Safety factor | 1.5x operating pressure |
| Inspection interval | 12 months or 2,000 operating hours, whichever comes first |
| Replacement cycle | 8 years or 16,000 operating hours |

> **CRITICAL NOTE:** Valve-B units that have exceeded their replacement cycle must be taken offline immediately. Continued operation above 2,800 psi on a cycle-expired Valve-B unit constitutes a Tier-3 safety event under SOP OP-204.

### 3.2 Hydraulic Valve-C (Secondary Lifting Circuit)

| Parameter | Specification |
|-----------|--------------|
| Maximum operating pressure | **3,200 psi** (221 bar) |
| Maximum allowable test pressure | 4,800 psi (331 bar) — do NOT exceed |
| Manufacturer | Atlas Hydraulic Systems, Model AHS-4400V-SC |
| Safety factor | 1.5x operating pressure |
| Inspection interval | 12 months or 2,000 operating hours, whichever comes first |

## 4. Supply Chain Risk Warnings

### 4.1 Sole Source Components

The following components have no qualified alternate supplier and represent single points of failure:
- **H2-NEX-007** (Emergency stop system) — SafetyFirst Industries (sole source)
- **H2-NEX-003** (Control system) — AVL Software Division (in-house only)

### 4.2 EMEA-Specific Supply Constraints

EuroVolt GmbH (alternate supplier for H2-NEX-008 power distribution units) is the ONLY qualified alternate supplier for the EMEA region. In the event of supply disruption, procurement must escalate to the Regional Operations Director per FIN-AUTH-101 Section 3.

## 5. Reference Documents

- **OP-204** — Incident Management Procedure (see operations/OP-204_Incident_Management_v4.md)
- **FIN-AUTH-101** — Finance Authorization Framework (see finance/FIN-AUTH-101_Authorization_Framework.md)
- **AVL-PUR-012** — Supplier Qualification and Management Policy (see supply_chain/AVL-PUR-012_Supplier_Qualification.md)

---

*Document control: This is a controlled document. Technical specifications are subject to revision. Reference the AVL DMS under document ID BOM-FLT-H2NEX for the latest revision.*
