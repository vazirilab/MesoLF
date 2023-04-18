function [bool] = is_legal_pathname(str)
bool = true;
try
    java.io.File(str).toPath;
catch
    bool = false;
end
end
