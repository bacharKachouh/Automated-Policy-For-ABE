import re

def parse_policy_to_abe_format(policy_text, attributes_prefix="hospitalA"):
    # Normalize
    policy_text = policy_text.lower()

    # Extract roles
    roles = []
    if "doctor" in policy_text:
        roles.append(f"{attributes_prefix}.doctor")
    if "researcher" in policy_text:
        roles.append(f"{attributes_prefix}.researcher")
    if "nurse" in policy_text:
        roles.append(f"{attributes_prefix}.nurse")
    if "admin" in policy_text:
        roles.append(f"{attributes_prefix}.admin")

    # Join roles with OR
    role_expr = " OR ".join(roles)
    if role_expr:
        role_expr = f"({role_expr})"

    # Extract department
    department = None
    for dept in ["cardiology", "oncology", "pharmacy", "emergency"]:
        if dept in policy_text:
            department = f"{attributes_prefix}.{dept}"
            break

    # Extract clearance level
    clearance = None
    match = re.search(r"clearance level (\d+)", policy_text)
    if match:
        level = match.group(1)
        clearance = f"{attributes_prefix}.clearance_level_{level}"

    # Compose full policy string
    components = [expr for expr in [role_expr, department, clearance] if expr]
    return " AND ".join(components)
policy_text = "Access Details:  This data can be accessed by users having roles: (Doctor or Researcher) and belonging to the Oncology department with clearance level 3."

policy_str = parse_policy_to_abe_format(policy_text)
print(policy_str)

