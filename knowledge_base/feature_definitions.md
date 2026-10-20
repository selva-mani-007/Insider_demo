# CERT r4.2 Extended Isolation Forest — Feature Definitions

Source: Project feature definitions from the CERT r4.2 notebook. These are operational definitions based on the feature-engineering code; interpret the features in that context.

## logon_count
Number of logon events for the user in the seven-day behavioural window.

## logon_ah
Count of logon events classified as after-hours. The notebook defines after-hours as before 08:00 or at/after 18:00.

## logon_we
Count of logon events occurring on Saturday or Sunday.

## unique_pcs
Number of distinct PCs observed for the user during the window.

## logon_ah_ratio
After-hours logons divided by total logons, using a denominator of at least one.

## usb_connects
Count of device events whose activity is classified as connect.

## usb_ah
Count of device events classified as after-hours.

## usb_we
Count of device events occurring on weekends.

## file_count
Number of file events in the window.

## file_ah
Count of file events classified as after-hours.

## file_we
Count of file events occurring on weekends.

## email_count
Number of email events in the window.

## email_ah
Count of email events classified as after-hours.

## email_size_total
Sum of the email size field for email events in the window. This is the dataset's size field, not a count of attachments or proof of exfiltration.

## http_count
Number of HTTP/web events in the window.

## job_visits
Count of web events whose extracted domain matches the notebook's configured job-site keyword list.

## threat_visits
Count of web events whose domain matches the notebook's configured threat-related keyword list. This is a heuristic configured-site category, not a finding that the destination is inherently malicious.

## http_ah
Count of web events classified as after-hours.

## unique_domains
Number of distinct extracted web domains observed in the window.

## exfil_overlap_days
Number of days in the window on which the user had at least one USB connection and at least one visit to a site classified by the notebook as a job or threat-related external site. Co-occurrence does not prove data transfer or event sequence.

## usb_per_logon
USB connections divided by logon count, using a denominator of at least one. It is a ratio, not a probability.

## ah_combined_score
Mean of five within-window after-hours activity ratios: logon, USB, file, email, and web. Each ratio uses a denominator of at least one. It summarizes the coded after-hours proportions across sources.

## job_usb_interaction
Product of job_visits and usb_connects in the same seven-day window. It is an engineered interaction value, not evidence that a job-site visit caused USB activity.
