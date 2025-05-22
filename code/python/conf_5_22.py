# DOCUMENT CONFIRMATION TRAINING GUIDE

You are reviewing documents to determine if they belong to a specific customer in our System of Record (SoR). Follow these guidelines to make accurate determinations.

Document:
"""
{row["full_text"].strip()}
"""

System of Record:
First Name: {row["First Name"]}
Last Name: {row["Last Name"]}
Address: {row["Mailing Address Line 1"]}

## DECISION RULE

Return confirmation as YES, IF:
- First Name AND Last Name are matches
OR
- Mailing Address Line 1 matches

Where "match" = exact match or partial match

## MATCHING DEFINITIONS

### Name Matching (Exact or Partial)
**Exact Match Examples:**
- Robert = Robert
- Johnson = Johnson

**Partial Match Examples:**
- Robert = Rob, Bob, Bobby
- William = Bill, Will, Billy
- James = Jim, Jimmy
- Elizabeth = Beth, Betty, Liz
- Charles = Chuck, Charlie
- Michael = Mike, Micky
- Jennifer = Jen, Jenny
- Christopher = Chris
- Matthew = Matt
- Katherine = Kate, Katie, Kathy

### Address Matching (Exact or Partial)
**Exact Match Examples:**
- 123 Main Street = 123 Main Street

**Partial Match Examples:**
- 123 Main Street = 123 Main St
- 456 Oak Avenue = 456 Oak Ave
- 789 Pine Road = 789 Pine Rd
- 101 First Boulevard = 101 1st Blvd
- Apartment 4B = Apt 4B = #4B = Unit 4B

**Shortened Address Examples:**
- 123 Main Street, Tampa, FL 33602 = 123 Main St
- 456 Oak Avenue, Unit 2A = 456 Oak Ave
- 789 Pine Road, Apartment B = 789 Pine Rd

## EVALUATION PROCESS

1. Check if First Name matches (exact or partial variation)
2. Check if Last Name matches (exact or partial variation)
3. If BOTH First Name AND Last Name match → Decision: YES
4. If names don't match, check Mailing Address Line 1
5. If Address matches (exact or partial) → Decision: YES
6. If neither condition is met → Decision: NO

## EXAMPLE SCENARIOS

**SCENARIO 1:**
SoR: Robert Johnson, 123 Oak Street
Document: "Rob Johnson at 456 Pine Ave"
Analysis: First Name match (Robert=Rob), Last Name match (Johnson=Johnson)
Decision: YES

**SCENARIO 2:**
SoR: Mary Williams, 456 Pine Avenue  
Document: "John Smith at 456 Pine Ave"
Analysis: Names don't match, but Address matches (456 Pine Avenue = 456 Pine Ave)
Decision: YES

**SCENARIO 3:**
SoR: Thomas Anderson, 789 Maple Drive
Document: "Tom Wilson at 123 Oak Street"
Analysis: Names don't match (Thomas=Tom but Anderson≠Wilson), Address doesn't match
Decision: NO

**SCENARIO 4:**
SoR: Jennifer Lopez, 101 Beach Road
Document: "Jenny Lopez at 202 Lake Drive"
Analysis: First Name match (Jennifer=Jenny), Last Name match (Lopez=Lopez)
Decision: YES

## RESPONSE FORMAT

After your analysis, provide your decision in this JSON format:
{
  "decision": "yes" or "no",
  "confidence": [numerical score 0-100],
  "explanation": "brief justification of your decision",
  "matching_elements": ["list what matched"],
  "non_matching_elements": ["list what didn't match"]
}
