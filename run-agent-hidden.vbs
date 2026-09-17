' Launches run-agent.cmd with no console window (Task Scheduler points here).
Dim fso, shell, here
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
shell.Run """" & here & "\run-agent.cmd""", 0, True
