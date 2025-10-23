# MIDI parser validator
Howdy partner! If you have stumbled unto this file, it means you are checking the entire codebase of this project. By this point you might be wondering, why the hell are these scripts so freaking long? (~17k loc total) and, what is their purpose?

Well, when i was first developing this project, i was very skeptical of it's correct behaviour, so i ended up writing a lot of scripts to help me verify all of its inner features, including many Miditok's parameters.

### Is this module overkill?
Undoubtedly, the parser itself is only about 4k lines of code, and the validator scripts are 17k lines long. It is a very long pipeline that i used at the very beginning of my project to find why my code wasn't working, now that it works, these scripts are not as necessary.

Feel free to modify any of these scripts to your liking, including the parser itself.