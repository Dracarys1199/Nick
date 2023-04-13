<?php
function encrypt($text, $s)
{
$result = "";
for ($i = 0; $i < strlen($text); $i++)
{

if (ctype_upper($text[$i]))
$result = $result.chr((ord($text[$i]) +$s - 65) % 26 + 65);

else
$result = $result.chr((ord($text[$i]) +$s - 97) % 26 + 97);

}

return $result;
}

$text = "Wonderful";
$s = 2;
echo "\nText :". $text;
echo "\nShift:". $s;
echo "\nCipher:". encrypt($text, $s);
?>