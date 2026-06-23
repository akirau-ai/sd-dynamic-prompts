(function() {
    function getPromptTextarea(textboxId) {
        return gradioApp().querySelector("#" + textboxId + " > label > textarea");
    }

    function getPromptTextbox(textboxId) {
        return gradioApp().getElementById(textboxId);
    }

    function getWarningElement(textboxId) {
        var textbox = getPromptTextbox(textboxId);
        if (!textbox) {
            return null;
        }

        var warningId = textboxId + "_sddp_conditional_prompt_warning";
        var warning = gradioApp().getElementById(warningId);

        if (!warning) {
            warning = document.createElement("div");
            warning.id = warningId;
            warning.className = "sddp-conditional-prompt-warning";
            textbox.appendChild(warning);
        }

        return warning;
    }

    function setWarning(textboxId, message) {
        var warning = getWarningElement(textboxId);
        if (!warning) {
            return;
        }

        warning.textContent = message || "";
        warning.style.display = message ? "block" : "none";
    }

    function splitAndConditions(condition) {
        return String(condition || "").split(" AND ").map(function(part) {
            return part.trim();
        });
    }

    function collectAssignedVariables(text) {
        var pattern = /\$\{\s*([A-Za-z_-][A-Za-z0-9_-]*)\s*=/g;
        var variables = {};
        var match;

        while ((match = pattern.exec(String(text || ""))) !== null) {
            variables[match[1]] = true;
        }

        return variables;
    }

    function isValidComparison(part) {
        return /^\s*[A-Za-z_-][A-Za-z0-9_-]*\s*(==|!=|\*=|!\*=)\s*.+\s*$/.test(part);
    }

    function parseComparison(part) {
        return /^\s*([A-Za-z_-][A-Za-z0-9_-]*)\s*(==|!=|\*=|!\*=)\s*(.*?)\s*$/.exec(part);
    }

    function validateCondition(condition, variables) {
        var parts = splitAndConditions(condition);
        if (!parts.length || parts.some(function(part) { return !part; })) {
            return "CP: bad condition";
        }

        for (var i = 0; i < parts.length; i += 1) {
            if (!isValidComparison(parts[i])) {
                return "CP: bad condition";
            }

            if (variables) {
                var comparison = parseComparison(parts[i]);
                if (!comparison) {
                    return "CP: bad condition";
                }

                var left = comparison[1].trim();
                var right = comparison[3].trim();
                if (!variables[left] && !variables[right]) {
                    return "CP: undefined var";
                }
            }
        }

        return "";
    }

    function parseBranches(body, variables) {
        var markerPattern = /\$\{elsif\s+([^}]+)\}|\$\{else\}/g;
        var seenElse = false;
        var match;

        while ((match = markerPattern.exec(body)) !== null) {
            var isElse = match[0] === "${else}";
            if (seenElse) {
                return "CP: else must be last";
            }
            if (isElse) {
                seenElse = true;
                continue;
            }

            var conditionError = validateCondition(match[1], variables);
            if (conditionError) {
                return conditionError;
            }
        }

        return "";
    }

    function validateConditionalPrompt(promptText) {
        var text = String(promptText || "");
        if (text.indexOf("${if ") === -1 && text.indexOf("${elsif ") === -1 && text.indexOf("${else}") === -1 && text.indexOf("${endif}") === -1) {
            return "";
        }

        var variables = collectAssignedVariables(text);
        var blockPattern = /\$\{if\s+([^}]+)\}([\s\S]*?)\$\{endif\}/g;
        var stripped = text.replace(blockPattern, function(_whole, ifCondition, body) {
            var conditionError = validateCondition(ifCondition, variables);
            if (conditionError) {
                throw new Error(conditionError);
            }

            var branchError = parseBranches(body, variables);
            if (branchError) {
                throw new Error(branchError);
            }

            return "";
        });

        if (stripped.indexOf("${if ") !== -1) {
            return "CP: missing ${endif}";
        }

        if (stripped.indexOf("${elsif ") !== -1 || stripped.indexOf("${else}") !== -1 || stripped.indexOf("${endif}") !== -1) {
            return "CP: broken block";
        }

        return "";
    }

    function safeValidate(promptText) {
        try {
            return validateConditionalPrompt(promptText);
        } catch (error) {
            return error && error.message ? error.message : "CP: bad syntax";
        }
    }

    function validatePrompt(textboxId) {
        var textarea = getPromptTextarea(textboxId);
        if (!textarea) {
            return;
        }

        setWarning(textboxId, safeValidate(textarea.value || ""));
    }

    function bindPrompt(textboxId) {
        var textarea = getPromptTextarea(textboxId);
        if (!textarea || textarea.dataset.sddpConditionalPromptWarningBound === "true") {
            return;
        }

        textarea.dataset.sddpConditionalPromptWarningBound = "true";

        textarea.addEventListener("input", function() {
            setTimeout(function() {
                validatePrompt(textboxId);
            }, 0);
        });

        setTimeout(function() {
            validatePrompt(textboxId);
        }, 0);
    }

    function setup() {
        bindPrompt("txt2img_prompt");
        bindPrompt("txt2img_neg_prompt");
        bindPrompt("img2img_prompt");
        bindPrompt("img2img_neg_prompt");
        bindPrompt("hires_prompt");
        bindPrompt("hires_neg_prompt");
    }

    onUiLoaded(setup);
    onAfterUiUpdate(setup);
})();
