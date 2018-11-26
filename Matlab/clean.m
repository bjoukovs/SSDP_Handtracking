function out = clean(in)

    ar = splitlines(in);
    
    out = [];
    
    for i=1:length(ar)-1
        if ~contains(ar(i), ';')
           temp = [];
           ar2 = split(ar(i),',');
           for j=1:length(ar2)
              temp = [temp str2double(ar2(j))]; 
           end
           out = [out; temp];
        end
    end

end

