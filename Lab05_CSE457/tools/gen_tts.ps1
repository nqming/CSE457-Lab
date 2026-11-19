# Sinh tiếng nói thô bằng Windows SAPI (David, Zira) -> lab5_data/raw
Add-Type -AssemblyName System.Speech
$out = Join-Path $PSScriptRoot "..\lab5_data\raw"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$sent = @(
 "The quick brown fox jumps over the lazy dog near the river bank.",
 "Please call me tomorrow morning before the meeting starts.",
 "Speech processing helps computers understand the human voice.",
 "She sells sea shells by the sea shore every summer.",
 "My favourite season is autumn because the weather is cool.",
 "The library opens at eight and closes at nine in the evening.",
 "We need to finish the project report by next Friday.",
 "Turn left at the second traffic light and go straight ahead.",
 "A cup of hot coffee makes every morning a little better.",
 "Signals can be analysed in both time and frequency domains."
)
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($v in @("Microsoft David Desktop","Microsoft Zira Desktop")) {
  $tag = if ($v -like "*David*") {"david"} else {"zira"}
  for ($i=0; $i -lt $sent.Count; $i++) {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice($v); $s.Rate = ($i % 3) - 1
    $s.SetOutputToWaveFile((Join-Path $out ("{0}_{1:D2}.wav" -f $tag,$i)), $fmt)
    $s.Speak($sent[$i]); $s.Dispose()
  }
}
"done"
