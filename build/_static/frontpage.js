
$(document).ready(function () {

    // hide prerequisite for the front-page
    $('.admonition-prerequisite').each(function(){ $(this).hide(); });

    // card
    $('.mx-card').each(function(){
        $(this).addClass('mdl-card mdl-shadow--2dp');
    });
    $('.mx-card .card-title').each(function(){
        $(this).addClass('mdl-card__title');
    });
    $('.mx-card .card-text').each(function(){
        $(this).addClass('mdl-card__supporting-text');
    });
    $('.card-link').each(function(){
        $(this).hide();
    });


    $('.mdl-card').each(function(){
        $(this).click(function() {
            window.location = $(this).find('.card-link').text();
            return true;
        });
    });
});
