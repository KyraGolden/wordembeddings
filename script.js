//siehe https://api.jquery.com/jQuery.post/
//https://api.jquery.com/jquery.ajax/

$(document).ready(function(){
    $("#getSynonyms").submit(function(){
        // Stop form from submitting normally
         event.preventDefault();
        // Get the values from elements on the page:
        var $form = $(this),
            inputword = $form.find("input[name='w']").val(),
            nr = $("#nr").children("option:selected").val(),
            radioValue = $("input[name='searchType']:checked").val();
        jQuery.ajax({
            url: "var/www/cgi-bin/get_synonyms.py",
            type: "POST",
            //contentType: "application/json",
            data: {w: inputword,
                   nr: nr,
                   st: radioValue
                   },
            success: function(response){
                console.log(response);
                $("#result").empty().append('<br>');
                var content = nr + " synonyms for ";
                $("#result").append(content);
                $("#result").append(inputword);
                $("#result").append(": <br>");
                $("#result").append(response.syn1);
                // oder:
                //alt_resp =  "<h5>Synonyms: " + resp + " </h5>";
                //return alt_resp;
            }
            //error: function(xhr, status, error){
            //    console.log(error)
            //}
        });
    });
});
