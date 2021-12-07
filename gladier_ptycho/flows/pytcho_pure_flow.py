flow_definition = {
  "Comment": "An analysis flow",
  "StartAt": "Transfer",
  "States": {
    "Transfer": {
      "Comment": "Initial transfer",
      "Type": "Action",
      "ActionUrl": "https://actions.automate.globus.org/transfer/transfer",
      "ActionScope": "https://auth.globus.org/scopes/actions.globus.org/transfer/transfer",
      "Parameters": {
        "source_endpoint_id.$": "$.input.source_endpoint", 
        "destination_endpoint_id.$": "$.input.dest_endpoint",
        "transfer_items": [
          {
            "source_path.$": "$.input.source_path",
            "destination_path.$": "$.input.dest_path",
            "recursive": True
          }
        ]
      },
      "ResultPath": "$.Transfer1Result",
      "WaitTime": 1800,
      "Next": "Analyze"
    },
    "Analyze": {
      "Comment": "Run a funcX function",
      "Type": "Action",
      "ActionUrl": "https://api.funcx.org/automate",
      "ActionScope": "https://auth.globus.org/scopes/facd7ccc-c5f4-42aa-916b-a0e270e2c2a9/automate2",
      "Parameters": {
          "tasks": [{
            "endpoint.$": "$.input.fx_ep",
            "func.$": "$.input.fx_id",
            "payload.$": "$.input.params"
        }]
      },
      "ResultPath": "$.AnalyzeResult",
      "WaitTime": 3600,
      "Next": "Transfer2"
    },
    "Transfer2": {
      "Comment": "Return transfer",
      "Type": "Action",
      "ActionUrl": "https://actions.automate.globus.org/transfer/transfer",
      "ActionScope": "https://auth.globus.org/scopes/actions.globus.org/transfer/transfer",
      "Parameters": {
        "source_endpoint_id.$": "$.input.dest_endpoint", 
        "destination_endpoint_id.$": "$.input.source_endpoint",
        "transfer_items": [
          {
            "source_path.$": "$.input.result_path",
            "destination_path.$": "$.input.source_result_path",
            "recursive": True #False
          }
        ]
      },
      "ResultPath": "$.Transfer2Result",
      "WaitTime": 1800,
      "End": True
    },
  }
}