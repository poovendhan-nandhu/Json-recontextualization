"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Loader2, CheckCircle2, XCircle, AlertCircle, FileText } from "lucide-react"

type LogEntry = {
  type: string
  message?: string
  stage?: string
  score?: number
  issues?: number
  passed?: boolean
  iteration?: number
  agent_scores?: Record<string, number>
  data?: any
}

export default function Home() {
  const [apiUrl, setApiUrl] = useState(() => {
    if (typeof window !== "undefined") {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
      return `${protocol}//${window.location.host}`
    }
    return "ws://localhost:8000"
  })
  const [scenarioPrompt, setScenarioPrompt] = useState("")
  const [inputJson, setInputJson] = useState("")
  const [isRunning, setIsRunning] = useState(false)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [finalResult, setFinalResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [validationReport, setValidationReport] = useState<string | null>(null)
  const [isGeneratingReport, setIsGeneratingReport] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  const getHttpUrl = () => {
    // Convert ws:// to http:// or wss:// to https://
    return apiUrl.replace("wss://", "https://").replace("ws://", "http://")
  }

  const runPipeline = () => {
    if (!scenarioPrompt.trim()) {
      setError("Please enter a scenario prompt")
      return
    }
    if (!inputJson.trim()) {
      setError("Please enter input JSON")
      return
    }

    let parsedJson: any
    try {
      parsedJson = JSON.parse(inputJson)
    } catch (e) {
      setError("Invalid JSON format")
      return
    }

    setError(null)
    setLogs([])
    setFinalResult(null)
    setValidationReport(null)
    setIsRunning(true)

    const wsUrl = `${apiUrl}/api/v1/pipeline/ws`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setLogs(prev => [...prev, { type: "info", message: "Connected to server" }])
      ws.send(JSON.stringify({
        input_json: parsedJson,
        scenario_prompt: scenarioPrompt
      }))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setLogs(prev => [...prev, data])

      if (data.type === "result") {
        setFinalResult(data)
        setIsRunning(false)
      } else if (data.type === "error") {
        setError(data.message)
        setIsRunning(false)
      }
    }

    ws.onerror = () => {
      setError("WebSocket connection failed")
      setIsRunning(false)
    }

    ws.onclose = () => {
      setIsRunning(false)
    }
  }

  const stopPipeline = () => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsRunning(false)
  }

  const generateReport = async () => {
    if (!finalResult?.data?.agent_scores) {
      setError("No validation data available for report")
      return
    }

    setIsGeneratingReport(true)
    setError(null)

    try {
      // Build agent_results from agent_scores for the report endpoint
      const agentResults = Object.entries(finalResult.data.agent_scores).map(([name, score]) => ({
        agent_name: name,
        score: score,
        passed: (score as number) >= 0.95,
        issues: []
      }))

      const response = await fetch(`${getHttpUrl()}/api/v1/validation/report/from-results`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          validation_results: { agent_results: agentResults },
          original_scenario: "Source Scenario",
          target_scenario: scenarioPrompt,
          simulation_purpose: "Business Simulation Training",
          output_format: "markdown"
        })
      })

      const data = await response.json()
      if (data.status === "success") {
        setValidationReport(data.report)
      } else {
        setError("Failed to generate report")
      }
    } catch (e) {
      setError(`Report generation failed: ${e}`)
    } finally {
      setIsGeneratingReport(false)
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.95) return "success"
    if (score >= 0.8) return "warning"
    return "destructive"
  }

  const downloadResult = () => {
    if (!finalResult?.data?.final_json) return
    const blob = new Blob([JSON.stringify(finalResult.data.final_json, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "adapted_output.json"
    a.click()
  }

  return (
    <main className="container mx-auto p-4 max-w-6xl">
      <h1 className="text-3xl font-bold mb-6 text-center">Cartedo Pipeline</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Input</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">API URL</label>
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="w-full px-3 py-2 border rounded-md text-sm"
                placeholder="ws://localhost:8000"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Scenario Prompt</label>
              <Textarea
                value={scenarioPrompt}
                onChange={(e) => setScenarioPrompt(e.target.value)}
                placeholder="Learners will act as junior consultants for a sustainable fashion brand..."
                className="h-24"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Input JSON</label>
              <Textarea
                value={inputJson}
                onChange={(e) => setInputJson(e.target.value)}
                placeholder='{"topicWizardData": {...}}'
                className="h-48 font-mono text-xs"
              />
            </div>
            <div className="flex gap-2">
              <Button onClick={runPipeline} disabled={isRunning} className="flex-1">
                {isRunning ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Running...
                  </>
                ) : (
                  "Run Pipeline"
                )}
              </Button>
              {isRunning && (
                <Button variant="outline" onClick={stopPipeline}>
                  Stop
                </Button>
              )}
            </div>
            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
                {error}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Logs Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Pipeline Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-96 overflow-y-auto border rounded-md p-3 bg-gray-50 font-mono text-xs space-y-1">
              {logs.length === 0 ? (
                <p className="text-gray-400">Pipeline logs will appear here...</p>
              ) : (
                logs.map((log, i) => (
                  <div key={i} className="flex items-start gap-2">
                    {log.type === "connected" && (
                      <span className="text-green-600">CONNECTED</span>
                    )}
                    {log.type === "stage_start" && (
                      <span className="text-blue-600">START: {log.stage}</span>
                    )}
                    {log.type === "stage_complete" && (
                      <span className="text-green-600">DONE: {log.stage}</span>
                    )}
                    {log.type === "validation_result" && (
                      <span className={log.passed ? "text-green-600" : "text-yellow-600"}>
                        VALIDATE [{log.iteration}]: {((log.score || 0) * 100).toFixed(1)}% ({log.issues} issues)
                      </span>
                    )}
                    {log.type === "result" && (
                      <span className="text-green-700 font-bold">COMPLETE</span>
                    )}
                    {log.type === "error" && (
                      <span className="text-red-600">ERROR: {log.message}</span>
                    )}
                    {log.type === "info" && (
                      <span className="text-gray-600">{log.message}</span>
                    )}
                    {!["connected", "stage_start", "stage_complete", "validation_result", "result", "error", "info"].includes(log.type) && (
                      <span className="text-gray-500">{log.type}: {log.message || JSON.stringify(log.data || {}).slice(0, 100)}</span>
                    )}
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Results Section */}
      {finalResult && (
        <Card className="mb-4">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              Results
              {finalResult.data?.validation_passed ? (
                <Badge variant="success">PASSED</Badge>
              ) : (
                <Badge variant="destructive">FAILED</Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
              <div className="text-center p-3 border rounded-md">
                <div className="text-2xl font-bold">
                  {((finalResult.data?.final_score || 0) * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-500">Final Score</div>
              </div>
              <div className="text-center p-3 border rounded-md">
                <div className="text-2xl font-bold">
                  {finalResult.data?.repair_iterations || 0}
                </div>
                <div className="text-sm text-gray-500">Repair Iterations</div>
              </div>
              <div className="text-center p-3 border rounded-md">
                <div className="text-2xl font-bold flex items-center justify-center">
                  {finalResult.data?.validation_passed ? (
                    <CheckCircle2 className="h-8 w-8 text-green-600" />
                  ) : (
                    <XCircle className="h-8 w-8 text-red-600" />
                  )}
                </div>
                <div className="text-sm text-gray-500">Status</div>
              </div>
              <div className="text-center p-3 border rounded-md">
                <Button size="sm" onClick={downloadResult} disabled={!finalResult.data?.final_json}>
                  Download JSON
                </Button>
              </div>
              <div className="text-center p-3 border rounded-md">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={generateReport}
                  disabled={isGeneratingReport}
                >
                  {isGeneratingReport ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      <FileText className="h-4 w-4 mr-1" />
                      Get Report
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Agent Scores */}
            {finalResult.data?.agent_scores && Object.keys(finalResult.data.agent_scores).length > 0 && (
              <div>
                <h3 className="font-semibold mb-2">Validator Scores</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {Object.entries(finalResult.data.agent_scores).map(([agent, score]) => (
                    <div key={agent} className="flex items-center justify-between p-2 border rounded text-sm">
                      <span className="truncate">{agent.replace(/_/g, " ")}</span>
                      <Badge variant={getScoreColor(score as number)}>
                        {((score as number) * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Validation Report Section */}
      {validationReport && (
        <Card className="mb-4">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Validation Report
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose prose-sm max-w-none bg-gray-50 p-4 rounded-md border overflow-auto max-h-[600px]">
              <pre className="whitespace-pre-wrap text-xs font-mono">{validationReport}</pre>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Decision Summary */}
      {finalResult && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Decision Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4 p-4 border rounded-md">
              {finalResult.data?.validation_passed ? (
                <>
                  <CheckCircle2 className="h-12 w-12 text-green-600" />
                  <div>
                    <h3 className="text-xl font-bold text-green-700">Acceptable</h3>
                    <p className="text-gray-600">
                      The adaptation passed all validation checks with a score of{" "}
                      {((finalResult.data?.final_score || 0) * 100).toFixed(1)}%.
                      Ready for review or deployment.
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <AlertCircle className="h-12 w-12 text-yellow-600" />
                  <div>
                    <h3 className="text-xl font-bold text-yellow-700">Needs Review</h3>
                    <p className="text-gray-600">
                      The adaptation scored {((finalResult.data?.final_score || 0) * 100).toFixed(1)}%
                      (target: 95%). Review the validator scores above for specific issues.
                    </p>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </main>
  )
}
