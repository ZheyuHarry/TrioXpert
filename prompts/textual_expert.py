####################################################
#      system prompts for textual analysis         #
####################################################
textual_system_prompt="""
## Role: textual expert
## Goal: 
- Integrate log abstraction and trace abstraction, and leverage key information for reasoning to accomplish downstream tasks in fault management.
## Constraints:
- The reasoning process must be based on log abstraction and trace abstraction.
- The reasoning process must ensure consistency from beginning to end.
- Strictly adhere to the output format specifications for each task.
## Instructions:
- You need to combine the descriptions of trace abstraction and log abstraction, and integrate key information to complete the fault analysis task.
- You need to accomplish the task based on the instructions and examples provided for each task.
"""

####################################################
#          prompts for Anomaly Detection           #
####################################################

ad_task_prompt="""
## Instructions:
- Analyze the log abstraction and trace abstraction, reason through them, and predict the time window most likely to experience a fault.
- Output the predicted time window wrapped in special tokens "<ad_time>" and "</ad_time>".
- Output the reasoning chains of the predicted time window and wrap it in special tokens "<Evidence>" and "</Evidence>".
## Example:
- Input: 
  - log abstraction: The logs indicate a series of errors occurring in the downstream service called by `webservice1` over a period of time. Specifically, there are 12 consecutive error logs from `webservice1` indicating that an error occurred in the downstream service, with the first error occurring at `1625715397143` and the last at `1625715975212`. This suggests a persistent issue with the downstream service, as `webservice1` continues to attempt to call it despite the repeated failures. The exact cause of the downstream service error remains unclear from the provided logs, but it could stem from issues such as network connectivity problems, incorrect endpoint configuration, or unavailability of the downstream service itself. Further investigation into the specific downstream service is necessary to pinpoint the exact cause.
  - trace abstraction: Input Traces:'9a6b3c7d8e9f0123': (1643723400000, 'ROOT', 'webservice1', 2100.45, True, 'http', 503, 'process_user_request'),(1643723400100, 'webservice1', 'logservice1', 1200.34, True, 'http', 500, 'log_user_activity'),(1643723400200, 'logservice1', 'logservice2', 800.76, False, 'http', 200, 'validate_log_data'),(1643723400150, 'webservice1', 'dbservice1', 1500.56, True, 'http', 503, 'fetch_user_profile'),(1643723400300, 'dbservice1', 'dbservice2', 1000.23, True, 'http', 200, 'process_user_data'),(1643723400250, 'dbservice1', 'dbservice3', 1200.45, False, 'http', 500, 'validate_user_data'),(1643723400350, 'dbservice2', 'dbservice4', 800.12, True, 'http', 200,'store_user_data'),(1643723400400, 'dbservice3', 'dbservice5', 1500.67, False, 'http', 500,'send_notification'),]## Output Abstraction:The provided traces reveal a complex microservice interaction with multiple dependencies and failure points. The root service (ROOT) initiates a request to webservice1, which subsequently calls logservice1 and dbservice1. While logservice1 successfully interacts with logservice2 (HTTP 200), dbservice1 encounters a critical issue. Specifically, dbservice1 fails when attempting to interact with dbservice3, leading it to return HTTP 503, which causes webservice1 to fail with the same status. The high latency observed in webservice1 (2100ms) and dbservice1 (1500ms) suggests that timeout thresholds may have been exceeded. Furthermore, the failure of dbservice3 to send a notification to dbservice5 (HTTP 500) indicates a potential issue with the notification mechanism. Potential causes include misconfigured timeout settings, insufficient error handling in dbservice1 and dbservice3, or a failure in the notification service. To address these issues
- output:
  - <ad_time>(1625715397143,1625715975212)</ad_time>
  - <Evidence>Log Time Span:First Error: `1625715397143` Last Error: `1625715975212` (duration: **approximately 9.6 minutes**).  Trace-Log Correlation**:  - The root service (ROOT) triggered a request in the trace, causing cascading failures in `webservice1`, with timing overlapping the log error window.  - Root cause: Failure of `dbservice1` calling `dbservice3` (HTTP 503), persisting throughout the log time range.  Error Propagation Path:  - High latency in `dbservice1` (1500ms) and notification failure from `dbservice3` to `dbservice5` (HTTP 500) exacerbated anomaly spread.</Evidence>
"""


####################################################
#          prompts for Root Cause Localization     #
####################################################

rcl_task_prompt="""
## Instructions:
- Analyze the log abstraction and trace abstraction, reason through them, and predict the top 5 root cause services that most likely to experience a fault.
- Output the predicted top 5 root cause services wrapped in special tokens "<rcl_rank>" and "</rcl_rank>".
- Output the reasoning chains of the predicted top 5 root cause services and wrap it in special tokens "<Evidence>" and "</Evidence>".
## Constraints:
- The predicted top 5 root cause services must be inside this list: ['IG01', 'IG02', 'MG01', 'MG02', 'Mysql01', 'Mysql02', 'Redis01', 'Redis02', 'Tomcat01', 'Tomcat02', 'Tomcat03', 'Tomcat04', 'apache01', 'apache02', 'dockerA1', 'dockerA2', 'dockerB1', 'dockerB2']
## Example:
- Input: 
  - log abstraction: The logs indicate a series of errors occurring in the downstream service called by `webservice1` over a period of time. Specifically, there are 12 consecutive error logs from `webservice1` indicating that an error occurred in the downstream service, with the first error occurring at `1625715397143` and the last at `1625715975212`. This suggests a persistent issue with the downstream service, as `webservice1` continues to attempt to call it despite the repeated failures. The exact cause of the downstream service error remains unclear from the provided logs, but it could stem from issues such as network connectivity problems, incorrect endpoint configuration, or unavailability of the downstream service itself. Further investigation into the specific downstream service is necessary to pinpoint the exact cause.
  - trace abstraction: Input Traces:'9a6b3c7d8e9f0123': (1643723400000, 'ROOT', 'webservice1', 2100.45, True, 'http', 503, 'process_user_request'),(1643723400100, 'webservice1', 'logservice1', 1200.34, True, 'http', 500, 'log_user_activity'),(1643723400200, 'logservice1', 'logservice2', 800.76, False, 'http', 200, 'validate_log_data'),(1643723400150, 'webservice1', 'dbservice1', 1500.56, True, 'http', 503, 'fetch_user_profile'),(1643723400300, 'dbservice1', 'dbservice2', 1000.23, True, 'http', 200, 'process_user_data'),(1643723400250, 'dbservice1', 'dbservice3', 1200.45, False, 'http', 500, 'validate_user_data'),(1643723400350, 'dbservice2', 'dbservice4', 800.12, True, 'http', 200,'store_user_data'),(1643723400400, 'dbservice3', 'dbservice5', 1500.67, False, 'http', 500,'send_notification'),]## Output Abstraction:The provided traces reveal a complex microservice interaction with multiple dependencies and failure points. The root service (ROOT) initiates a request to webservice1, which subsequently calls logservice1 and dbservice1. While logservice1 successfully interacts with logservice2 (HTTP 200), dbservice1 encounters a critical issue. Specifically, dbservice1 fails when attempting to interact with dbservice3, leading it to return HTTP 503, which causes webservice1 to fail with the same status. The high latency observed in webservice1 (2100ms) and dbservice1 (1500ms) suggests that timeout thresholds may have been exceeded. Furthermore, the failure of dbservice3 to send a notification to dbservice5 (HTTP 500) indicates a potential issue with the notification mechanism. Potential causes include misconfigured timeout settings, insufficient error handling in dbservice1 and dbservice3, or a failure in the notification service. To address these issues
- Output:
  - <rcl_rank>[dbservice1,dbservice3,webservice1,logservice1,dbservice2]</rcl_rank>
  - <Evidence>Root cause analysis of the system failure indicates that the core issue lies in an anomaly within the downstream database services in the microservice interaction chain. Log analysis reveals persistent errors in the downstream service called by webservice1 between timestamps 1625715397143 and 1625715975212, potentially involving network connectivity issues, configuration errors, or downstream service unavailability. Trace analysis of the call chain shows that after the root service (ROOT) initiates a request to webservice1, logservice1 successfully interacts with logservice2 (HTTP 200), while dbservice1 encounters a critical error when invoking dbservice3, returning HTTP 503. This failure propagates upward, causing webservice1 to fail with the same status. The root cause is isolated to the interaction failure between dbservice1 and dbservice3, warranting priority investigation into resource allocation, dependency issues, or code logic anomalies at this node.</Evidence>
"""

####################################################
#          prompts for Faliure Triage              #
####################################################
ft_task_prompt="""
## Instructions:
- Analyze the log abstraction and trace abstraction, reason through them, and predict the failure type of the failure.
- Output the predicted failure type wrapped in special tokens "<ft>" and "</ft>".
- Output the reasoning chains of the predicted top 5 root cause services and wrap it in special tokens "<Evidence>" and "</Evidence>".
## Constraints:
- The predicted failure type must be inside this list: ['MEM', 'JVM;CPU', 'JVM;MEM', 'CPU', 'Network', 'Disk']
## Example:
- Input: 
  - log abstraction: The logs indicate a series of errors occurring in the downstream service called by `webservice1` over a period of time. Specifically, there are 12 consecutive error logs from `webservice1` indicating that an error occurred in the downstream service, with the first error occurring at `1625715397143` and the last at `1625715975212`. This suggests a persistent issue with the downstream service, as `webservice1` continues to attempt to call it despite the repeated failures. The exact cause of the downstream service error remains unclear from the provided logs, but it could stem from issues such as network connectivity problems, incorrect endpoint configuration, or unavailability of the downstream service itself. Further investigation into the specific downstream service is necessary to pinpoint the exact cause.
  - trace abstraction: Input Traces:'9a6b3c7d8e9f0123': (1643723400000, 'ROOT', 'webservice1', 2100.45, True, 'http', 503, 'process_user_request'),(1643723400100, 'webservice1', 'logservice1', 1200.34, True, 'http', 500, 'log_user_activity'),(1643723400200, 'logservice1', 'logservice2', 800.76, False, 'http', 200, 'validate_log_data'),(1643723400150, 'webservice1', 'dbservice1', 1500.56, True, 'http', 503, 'fetch_user_profile'),(1643723400300, 'dbservice1', 'dbservice2', 1000.23, True, 'http', 200, 'process_user_data'),(1643723400250, 'dbservice1', 'dbservice3', 1200.45, False, 'http', 500, 'validate_user_data'),(1643723400350, 'dbservice2', 'dbservice4', 800.12, True, 'http', 200,'store_user_data'),(1643723400400, 'dbservice3', 'dbservice5', 1500.67, False, 'http', 500,'send_notification'),]## Output Abstraction:The provided traces reveal a complex microservice interaction with multiple dependencies and failure points. The root service (ROOT) initiates a request to webservice1, which subsequently calls logservice1 and dbservice1. While logservice1 successfully interacts with logservice2 (HTTP 200), dbservice1 encounters a critical issue. Specifically, dbservice1 fails when attempting to interact with dbservice3, leading it to return HTTP 503, which causes webservice1 to fail with the same status. The high latency observed in webservice1 (2100ms) and dbservice1 (1500ms) suggests that timeout thresholds may have been exceeded. Furthermore, the failure of dbservice3 to send a notification to dbservice5 (HTTP 500) indicates a potential issue with the notification mechanism. Potential causes include misconfigured timeout settings, insufficient error handling in dbservice1 and dbservice3, or a failure in the notification service. 
- Output:
  - <ft>['Container Hardware']</ft>
  - <Evidence>Root cause analysis of the system failure indicates that the core issue stems from memory resource exhaustion in containerized services, as evidenced by persistent HTTP 503 errors and cascading failures across dependent services. Log analysis reveals repeated downstream service errors from webservice1 over an extended period (12 consecutive failures between timestamps 1625715397143 and 1625715975212), suggesting critical resource constraints preventing successful request processing. Trace analysis highlights prolonged latency (e.g., 2100ms in webservice1, 1500ms in dbservice1) and failed interactions (HTTP 500/503) in the call chain, particularly during data validation and notification phases. These symptoms align with memory overload scenarios where containers exhaust allocated memory, leading to degraded performance, request timeouts, and process termination. The systemic propagation of failures across services further supports the hypothesis of memory-related bottlenecks disrupting critical workflows.</Evidence>
"""